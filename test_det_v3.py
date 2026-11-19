import copy
import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser
import types

import submitit
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf, open_dict

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def ensure_sam2_on_path():
    repo_root = os.path.abspath(os.path.dirname(__file__))
    sam2_root = os.path.join(repo_root, "sam2")
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)


def add_pythonpath_to_sys_path():
    sam2_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "sam2"))
    graco_path = os.path.join(sam2_root, "GraCo")

    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)
    if graco_path not in sys.path:
        sys.path.insert(0, graco_path)

    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


def _ensure_val_data(cfg):
    data_conf = cfg.trainer.get("data", None)
    if data_conf is None or data_conf.get("val", None) is not None:
        return
    if data_conf.get("train", None) is None:
        return
    with open_dict(cfg):
        data_conf = cfg.trainer.data
        data_conf.val = copy.deepcopy(data_conf.train)
        if hasattr(data_conf.val, "shuffle"):
            data_conf.val.shuffle = False
        if hasattr(data_conf.val, "drop_last"):
            data_conf.val.drop_last = False
        if hasattr(data_conf.val, "datasets"):
            for ds in data_conf.val.datasets:
                if hasattr(ds, "training"):
                    ds.training = False
        if hasattr(data_conf.val, "collate_fn") and hasattr(data_conf.val.collate_fn, "dict_key"):
            data_conf.val.collate_fn.dict_key = "val"

        if "loss" in cfg.trainer and cfg.trainer.loss is not None:
            loss_conf = cfg.trainer.loss
            if "val" not in loss_conf and "all" in loss_conf:
                loss_conf.val = copy.deepcopy(loss_conf.all)


def _prepare_cfg(args):
    cfg = compose(config_name=args.config)
    cfg.trainer.mode = args.mode
    if args.resume_from:
        with open_dict(cfg):
            if "checkpoint" not in cfg.trainer:
                cfg.trainer.checkpoint = {}
            cfg.trainer.checkpoint.resume_from = args.resume_from
    if args.img_folder or args.gt_folder:
        with open_dict(cfg):
            if args.img_folder:
                cfg.dataset.img_folder = args.img_folder
            if args.gt_folder:
                cfg.dataset.gt_folder = args.gt_folder

    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    if args.output_dir:
        cfg.launcher.experiment_log_dir = args.output_dir

    _ensure_val_data(cfg)
    return cfg


def _write_configs(cfg):
    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))
    return cfg_resolved


def apply_granularity_override(trainer, granularity_value):
    orig_step = trainer._step

    def _step_with_override(self, batch, model, phase):
        if getattr(batch, "granularities", None) is not None:
            batch.granularities = torch.full_like(batch.granularities, granularity_value)
        return orig_step(batch, model, phase)

    trainer._step = types.MethodType(_step_with_override, trainer)


def single_proc_run(local_rank, main_port, cfg, world_size):
    ensure_sam2_on_path()
    add_pythonpath_to_sys_path()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    if getattr(cfg, "granularity_override", None) is not None:
        apply_granularity_override(trainer, cfg.granularity_override)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method("spawn")
    if num_proc == 1:
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


class SubmititRunner(submitit.helpers.Checkpointable):
    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port

    def run_trainer(self):
        job_env = submitit.JobEnvironment()
        add_pythonpath_to_sys_path()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        if getattr(cfg_resolved, "granularity_override", None) is not None:
            apply_granularity_override(trainer, cfg_resolved.granularity_override)
        trainer.run()

    def __call__(self):
        try:
            self.run_trainer()
        except Exception as e:
            logging.error(format_exception(e))
            raise e


def main(args) -> None:
    ensure_sam2_on_path()
    cfg = _prepare_cfg(args)
    print("###################### Test App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("###########################################################")

    if args.granularity is not None:
        with open_dict(cfg):
            cfg.granularity_override = float(args.granularity)
    cfg_resolved = _write_configs(cfg)
    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = os.path.join(cfg.launcher.experiment_log_dir, "submitit_logs")

    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )

    if submitit_conf.use_cluster:
        executor = submitit.AutoExecutor(folder=submitit_dir)
        submitit_conf.partition = (
            args.partition
            if args.partition is not None
            else submitit_conf.get("partition", None)
        )
        submitit_conf.account = (
            args.account
            if args.account is not None
            else submitit_conf.get("account", None)
        )
        submitit_conf.qos = (
            args.qos if args.qos is not None else submitit_conf.get("qos", None)
        )
        job_kwargs = {
            "timeout_min": 60 * submitit_conf.timeout_hour,
            "name": submitit_conf.name if hasattr(submitit_conf, "name") else args.config,
            "slurm_partition": submitit_conf.partition,
            "gpus_per_node": cfg.launcher.gpus_per_node,
            "tasks_per_node": cfg.launcher.gpus_per_node,
            "cpus_per_task": submitit_conf.cpus_per_task,
            "nodes": cfg.launcher.num_nodes,
            "slurm_additional_parameters": {
                "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
            },
        }
        if "include_nodes" in submitit_conf:
            assert (
                len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes
            ), "Not enough nodes"
            job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
                submitit_conf["include_nodes"]
            )
        if submitit_conf.account is not None:
            job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
        if submitit_conf.qos is not None:
            job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos

        if submitit_conf.get("mem_gb", None) is not None:
            job_kwargs["mem_gb"] = submitit_conf.mem_gb
        elif submitit_conf.get("mem", None) is not None:
            job_kwargs["slurm_mem"] = submitit_conf.mem

        if submitit_conf.get("constraints", None) is not None:
            job_kwargs["slurm_constraint"] = submitit_conf.constraints

        if submitit_conf.get("comment", None) is not None:
            job_kwargs["slurm_comment"] = submitit_conf.comment

        if submitit_conf.get("srun_args", None) is not None:
            job_kwargs["slurm_srun_args"] = []
            if submitit_conf.srun_args.get("cpu_bind", None) is not None:
                job_kwargs["slurm_srun_args"].extend(
                    ["--cpu-bind", submitit_conf.srun_args.cpu_bind]
                )

        print("###################### SLURM Config ####################")
        print(job_kwargs)
        print("##########################################")
        executor.update_parameters(**job_kwargs)

        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        runner = SubmititRunner(main_port, cfg_resolved)
        job = executor.submit(runner)
        print(f"Submitit Job ID: {job.job_id}")
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(
            submitit_conf.port_range[0], submitit_conf.port_range[1]
        )
        single_node_runner(cfg_resolved, main_port)


if __name__ == "__main__":
    ensure_sam2_on_path()
    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="config name under sam2 (e.g. configs/unsamv2_small_training.yaml)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path to resume for evaluation",
    )
    parser.add_argument(
        "--img-folder",
        type=str,
        default=None,
        help="Override dataset.img_folder for evaluation",
    )
    parser.add_argument(
        "--gt-folder",
        type=str,
        default=None,
        help="Override dataset.gt_folder for evaluation",
    )
    parser.add_argument(
        "--granularity",
        type=float,
        default=None,
        help="Override granularity value for evaluation",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="val",
        choices=["val", "train", "train_only"],
        help="Trainer mode to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override experiment_log_dir",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    main(args)
