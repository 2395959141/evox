import dataclasses
import warnings
from functools import wraps, partial
from collections import namedtuple
from typing import Annotated, Any, Callable, Optional, Tuple, TypeVar, get_type_hints

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node, tree_map, tree_leaves


from .state import State


"""EvoX框架的核心模块管理系统
这个模块实现了EvoX中的状态管理机制，主要包含：
1. Stateful基类：所有EvoX模块的基础类
2. 状态管理装饰器：用于处理模块状态的自动提取和合并
3. JIT相关功能：用于优化计算性能
"""

def use_state(func: Callable, index: int = None):
    """状态管理装饰器
    
    在EvoX中，状态是不可变的，这个装饰器通过以下步骤实现状态管理：
    1. 从全局状态树中定位并提取当前模块的状态
    2. 执行模块的计算逻辑
    3. 将更新后的状态合并回状态树
    4. 支持批处理状态的索引访问（用于并行计算）
    
    参数:
        func: 需要管理状态的方法
        index: 批处理状态的索引，用于访问批处理状态中的特定样本
    """
    
    err_msg = "Expect last return value must be State, got {}"

    def wrapper(self, state: State, *args, **kwargs):
        # 验证输入的state参数类型必须是State类
        assert isinstance(
            state, State
        ), f"The first argument must be `State`, got {type(state)}"
        
        # 检查模块是否已经初始化(是否有_node_id和_module_name属性)
        if not hasattr(self, "_node_id") or not hasattr(self, "_module_name"):
            raise ValueError(
                f"{self} is not initialized, did you forget to call `init`?"
            )

        # 在状态树中查找当前模块对应的状态
        path, matched_state = state.find_path_to(self._node_id, self._module_name)

        # 处理批处理情况 - 如果指定了index,则提取对应批次的状态
        if index is not None:
            # 使用tree_map提取指定索引的状态和模块
            extracted_state = tree_map(lambda x: x[index], matched_state)
            this_module = tree_map(lambda x: x[index], self)
        else:
            # 否则使用完整状态
            extracted_state = matched_state
            this_module = self

        # 根据方法类型调用函数
        if hasattr(func, "__self__"):
            # 如果是绑定方法,不传入self
            return_value = func(extracted_state, *args, **kwargs)
        else:
            # 如果是未绑定方法(类方法),传入self
            return_value = func(this_module, extracted_state, *args, **kwargs)

        # 处理返回值 - 必须包含State对象
        if not isinstance(return_value, tuple):
            # 单个返回值必须是State类型
            assert isinstance(return_value, State), err_msg.format(type(return_value))
            aux, new_state = None, return_value
        else:
            # 多个返回值时,最后一个必须是State类型
            assert isinstance(return_value[-1], State), err_msg.format(
                type(return_value[-1])
            )
            # 解包返回值为辅助数据和新状态
            aux, new_state = return_value[:-1], return_value[-1]

        # 如果是批处理模式,将更新后的状态写回原始批处理数组
        if index is not None:
            new_state = tree_map(
                lambda batch_arr, new_arr: batch_arr.at[index].set(new_arr),
                matched_state,
                new_state,
            )

        # 使用新状态更新状态树
        state = state.replace_by_path(path, new_state)

        # 返回结果:如果没有辅助数据就只返回状态,否则返回辅助数据和状态
        if aux is None:
            return state
        else:
            return (*aux, state)

    if hasattr(func, "__self__"):
        return wraps(func)(partial(wrapper, func.__self__))
    else:
        return wraps(func)(wrapper)


def jit_method(method: Callable):
    """Decorator for methods, wrapper the method with jax.jit, and set self as static argument.

    Parameters
    ----------
    method
        A python method

    Returns
    -------
    function
        A jit wrapped version of this method
    """
    return jax.jit(
        method,
        static_argnums=[
            0,
        ],
    )


def default_jit_func(name: str):
    if name == "__call__":
        return True

    if name.startswith("_"):
        return False

    return True


def jit_class(cls):
    """用于将类的方法进行JIT编译的辅助装饰器函数
    
    工作流程：
    1. 遍历类的所有属性
    2. 对符合条件的方法应用JIT编译
    3. 返回优化后的类
    """
    # 遍历类的所有属性
    for attr_name in dir(cls):
        # 获取属性对象
        func = getattr(cls, attr_name)
        # 检查是否是可调用对象(方法)且满足JIT编译条件
        if callable(func) and default_jit_func(attr_name):
            # 根据类的类型选择不同的JIT包装方式
            if dataclasses.is_dataclass(cls):
                # 如果是数据类，直接使用jax.jit包装
                wrapped = jax.jit(func)
            else:
                # 如果是普通类，使用jit_method包装（会将self设为静态参数）
                wrapped = jit_method(func)
            # 将包装后的方法设置回类中
            setattr(cls, attr_name, wrapped)
    return cls


class Stateful:
    """EvoX框架中所有模块的基类
    
    设计特点：
    1. 分离不可变参数和可变状态：
       - __init__方法用于初始化不可变的超参数
       - setup方法用于初始化可变状态
    2. 支持模块嵌套：
       - 通过_recursive_init自动初始化所有子模块
       - 维护模块间的层次结构
    3. 状态管理：
       - 每个模块都有唯一的node_id和module_name
       - 支持状态的批处理操作
    """

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "_node_id", None)
        object.__setattr__(self, "_module_name", None)

    def setup(self, key: jax.Array) -> State:
        """初始化模块的可变状态
        
        注意：返回的State对象本身是不可变的，
        状态的更新通过返回新的State对象实现
        
        参数:
            key: JAX随机数生成器的密钥
        返回:
            包含模块初始状态的State对象
        """
        return State()

    def _recursive_init(
        self, key: jax.Array, node_id: int, module_name: str, no_state: bool
    ) -> Tuple[State, int]:
        """递归初始化模块及其所有子模块
        
        工作流程：
        1. 为当前模块分配唯一标识
        2. 收集并排序所有子模块（确保初始化顺序确定性）
        3. 递归初始化每个子模块
        4. 合并所有状态到状态树中
        
        参数:
            key: 随机数生成器密钥
            node_id: 模块的唯一标识符
            module_name: 模块名称
            no_state: 是否跳过状态初始化
        """
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_module_name", module_name)

        if not no_state:
            child_states = {}

        # Find all submodules and sort them according to their name.
        # Sorting is important because it makes sure that the node_id
        # is deterministic across different runs.
        SubmoduleInfo = namedtuple("Submodule", ["name", "module", "metadata"])

        submodules = []
        # preprocess and sort to make sure the order is deterministic
        # otherwise the node_id will be different across different runs
        # making save/load impossible
        if dataclasses.is_dataclass(self):
            for field in dataclasses.fields(self):
                attr = getattr(self, field.name)

                if isinstance(attr, Stateful):
                    submodules.append(SubmoduleInfo(field.name, attr, field.metadata))
        else:
            for attr_name in vars(self):
                attr = getattr(self, attr_name)
                if not attr_name.startswith("_") and isinstance(attr, Stateful):
                    submodules.append(SubmoduleInfo(attr_name, attr, {}))

        submodules.sort()

        for attr_name, attr, metadata in submodules:
            if key is None:
                subkey = None
            else:
                key, subkey = jax.random.split(key)

            # handle "StackAnnotation"
            # attr should be a list, or tuple of modules
            if metadata.get("stack", False):
                num_copies = len(attr)
                subkeys = jax.random.split(subkey, num_copies)
                current_node_id = node_id
                _, node_id = attr._recursive_init(None, node_id + 1, attr_name, True)
                submodule_state, _node_id = jax.vmap(
                    partial(
                        Stateful._recursive_init,
                        node_id=current_node_id + 1,
                        module_name=attr_name,
                        no_state=no_state,
                    )
                )(attr, subkeys)
            else:
                submodule_state, node_id = attr._recursive_init(
                    subkey, node_id + 1, attr_name, no_state
                )

            if not no_state:
                assert isinstance(
                    submodule_state, State
                ), "setup method must return a State"
                child_states[attr_name] = submodule_state
        if no_state:
            return None, node_id
        else:
            self_state = self.setup(key)
            if dataclasses.is_dataclass(self_state):
                # if the setup method return a dataclass, convert it to State first
                self_state = State.from_dataclass(self_state)

            self_state._set_state_id_mut(self._node_id)._set_child_states_mut(
                child_states
            ),
            return self_state, node_id

    def init(self, key: jax.Array = None, no_state: bool = False) -> State:
        """Initialize this module and all submodules

        This method should not be overwritten.

        Parameters
        ----------
        key
            A PRNGKey.

        Returns
        -------
        State
            The state of this module and all submodules combined.
        """
        state, _node_id = self._recursive_init(key, 0, None, no_state)
        return state

    @classmethod
    def stack(cls, stateful_objs, axis=0):
        for obj in stateful_objs:
            assert dataclasses.is_dataclass(obj), "All objects must be dataclasses"

        def stack_arrays(array, *arrays):
            return jnp.stack((array, *arrays), axis=axis)

        return tree_map(stack_arrays, stateful_objs[0], *stateful_objs[1:])

    def __len__(self) -> int:
        """
        Inspect the length of the first element in the state,
        usually paired with `Stateful.stack` to read the batch size
        """
        assert dataclasses.is_dataclass(self), "Length is only supported for dataclass"

        return len(tree_leaves(self)[0])
