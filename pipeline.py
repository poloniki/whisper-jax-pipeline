# sudo pip3 install kfp --pre
import kfp
from kfp import compiler

import kfp.dsl as dsl


@dsl.container_component
def whisper_jax():
    return dsl.ContainerSpec(
        image="europe-southwest1-docker.pkg.dev/wagon-bootcamp-355610/whisper-jax/pipeline-task:latest",
        args=[
            "--audio_file",
            "gs://taxifare_super_bucket/myfile.mp3",
            "--transcript_file",
            "gs://taxifare_super_bucket/example.txt",
        ],
    )


@dsl.pipeline(name="whisper-jax")
def transcribe_pipeline():
    task = whisper_jax()
    task.set_cpu_request("4")
    task.set_cpu_limit("4")
    task.set_memory_request("32Gi")
    task.set_memory_limit("32Gi")
    task.set_accelerator_type("NVIDIA_TESLA_T4")
    task.set_accelerator_limit(1)


compiler.Compiler().compile(transcribe_pipeline, "pipeline.json")
