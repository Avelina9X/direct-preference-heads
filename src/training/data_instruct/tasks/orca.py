from collections.abc import Callable, Mapping
from datasets import DatasetDict, Dataset, load_dataset

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class OpenOrcaInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'Open-Orca/OpenOrca', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED

    @property
    def task_name( self ) -> str:
        return 'orca'

    @property
    def task_subset( self ) -> str:
        return 'OpenOrca'

    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]

    def get_validation_docs( self ) -> None:
        return None

    def get_test_docs( self ) -> None:
        return None

    def get_fewshot_docs( self ) -> None:
        return None

    def format_system_message( self, doc: dict ) -> Message:
        noprompt = (
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.'
        )
        prompt = doc[ 'system_prompt' ]

        return Message(
            role='system',
            content=prompt if len( prompt ) > 0 else noprompt,
            complete=True,
        )

    def format_user_message( self, doc: dict ) -> Message:
        return Message(
            role='user',
            content=doc['question'],
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['response'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return []

    def format_unlabelled_messages( self, doc: dict ) -> list[Message]:
        return []

    def create_unlabelled_message_target( self, doc: dict ) -> None:
        return None

    def compute_metric( self, predictions=None, references=None ) -> dict:
        # TODO: add warning for using compute
        return {}

class OpenOrcaPairsInstructDataset( OpenOrcaInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        dataset = load_dataset( 'Intel/orca_dpo_pairs', cache_dir=cache_dir )
        assert isinstance( dataset, DatasetDict )
        return dataset

    @property
    def task_subset( self ) -> str:
        return 'orca_dpo_pairs'

    def format_system_message( self, doc: dict ) -> Message:
        noprompt = (
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.'
        )
        prompt = doc[ 'system' ]

        return Message(
            role='system',
            content=prompt if len( prompt ) > 0 else noprompt,
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['chosen'],
            complete=True,
        ) ]

    def format_distractor_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['rejected'],
            complete=True,
        ) ]


DIRECTORY: Mapping[str, Callable[[str], BaseInstructDataset]] = {
    'OpenOrca': OpenOrcaInstructDataset,
    'orca_dpo_pairs': OpenOrcaPairsInstructDataset,
}

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich

    cache_dir = os.environ[ 'HF_CACHE_DIR' ]

    rich.print( OpenOrcaInstructDataset( cache_dir ) )
