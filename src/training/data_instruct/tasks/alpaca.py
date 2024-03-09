from datasets import DatasetDict, Dataset, load_dataset
from evaluate import load as load_metric

from ..task_base import BaseInstructDataset, InstructionDatasetTask, Message

class AlpacaInstructDataset( BaseInstructDataset ):
    def download( self, cache_dir: str ) -> DatasetDict:
        return load_dataset( 'yahma/alpaca-cleaned', cache_dir=cache_dir )

    @property
    def task_type( self ) -> InstructionDatasetTask:
        return InstructionDatasetTask.INSTRUCT_CLOSED
    
    @property
    def group_name( self ) -> str | None:
        return None
    
    @property
    def task_name( self ) -> str:
        return 'alpaca'
    
    def get_training_docs( self ) -> Dataset:
        return self.dataset[ 'train' ]
    
    def get_validation_docs( self ) -> None:
        return None
    
    def get_test_docs( self ) -> None:
        return None
    
    def get_fewshot_docs( self ) -> None:
        return None
    

    def format_system_message( self, doc: dict ) -> Message:
        prompt_nocontext = (
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.'
        )
        prompt_context = (
            'Below is an instruction that describes a task, paired with an input that provides further context. '
            'Write a response that appropriately completes the request.'
        )

        return Message(
            role='system',
            content=prompt_context if len( doc['input'] ) > 0 else prompt_nocontext,
            complete=True,
        )

    def format_user_message( self, doc: dict ) -> Message:
        prompt = (
            f'Instruction:\n'
            f'{doc["instruction"]}\n'
            f'\n'
        )

        if len( doc['input'] ) > 0:
            prompt += (
                f'Input:\n'
                f'{doc["input"]}\n'
                f'\n'
            )

        prompt += (
            'Answer:'
        )
        
        return Message(
            role='user',
            content=prompt,
            complete=True,
        )

    def format_target_messages( self, doc: dict ) -> list[Message]:
        return [ Message(
            role='assistant',
            content=doc['output'],
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

def main():
    # pylint: disable=W0611
    # pylint: disable=C0415
    import os
    import rich
    
    cache_dir = os.environ[ 'HF_CACHE_DIR' ]
    
    rich.print( AlpacaInstructDataset( cache_dir ) )