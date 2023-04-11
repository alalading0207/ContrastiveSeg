import os

from lib.utils.tools.logger import Logger as Log
from . import standard

evaluators = {
    'standard': standard.StandardEvaluator
}


def get_evaluator(configer, trainer, name=None):
    name = os.environ.get('evaluator', 'standard')   # 获取环境变量

    if not name in evaluators:
        raise RuntimeError('Unknown evaluator name: {}'.format(name))
    
    klass = evaluators[name]   # evaluators[standard]
    Log.info('Using evaluator: {}'.format(klass.__name__))   # Using evaluator: StandardEvaluator

    return klass(configer, trainer)
