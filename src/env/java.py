import jpype
import jpype.imports
from jpype.types import *
from typing import List, Callable
from src.env.env import SimulationBase, IntervalResult
from contextlib import contextmanager


@contextmanager
def jvm_context(classpath: List[str]):
    """
    A context manager to manage the lifecycle of the JVM.
    :param classpath: List of paths or a single path string to the class files or JARs.
    """
    try:
        # Start JVM if not already started
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=classpath)
        yield
    finally:
        # Shutdown the JVM when the block exits
        if jpype.isJVMStarted():
            jpype.shutdownJVM()


class JavaClassManager:
    @classmethod
    def create_instance(cls, class_name, *args, **kwargs):
        """
        Class method to start the JVM, load a Java class, and return an instance of the class.
        :param classpath: Path to the directory containing Java class files or JARs.
        :param class_name: Fully qualified name of the Java class to load.
        :return: Instance of the Java class, or None if there was an error.
        """
        # Start the JVM
        if not jpype.isJVMStarted():
            raise RuntimeError("JVM is not started. Please start the JVM before loading Java classes. You can use the"
                               "jvm_context context manager to start the JVM to ensure proper cleanup.")

        # Load the Java class
        try:
            java_class = jpype.JClass(class_name)
            return java_class(*args)  # Return an instance of the Java class
        except Exception as e:
            print(f"Error loading Java class {class_name}: {e}")
            return None

    @staticmethod
    def shutdown_jvm():
        """
        Static method to shut down the JVM cleanly.
        """
        if jpype.isJVMStarted():
            jpype.shutdownJVM()


class SimulationWrapper(SimulationBase):
    def __init__(self, class_name: str, *args, **kwargs):
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs
        self.instance = JavaClassManager.create_instance(self.class_name, *self.args)

    def reset(self) -> IntervalResult:
        interval_result = self.instance.reset()
        convert_fn = lambda x: list(x)

        return IntervalResult(self._convert_java_double_map(interval_result.getCurrentSystemLatencies()),
                              self._convert_java_single_map(interval_result.getCurrentSystemConfiguration(),
                                                            convert_fn))

    def runInterval(self) -> IntervalResult:
        interval_result = self.instance.runInterval()

        convert_fn = lambda x: list(x)

        return IntervalResult(self._convert_java_double_map(interval_result.getCurrentSystemLatencies()),
                              self._convert_java_single_map(interval_result.getCurrentSystemConfiguration(),
                                                            convert_fn))

    def _convert_java_double_map(self, java_map: jpype.JObject):
        return {outer_key: {inner_key: inner_value for inner_key, inner_value in outer_value.entrySet()}
                for outer_key, outer_value in java_map.entrySet()}

    def _convert_java_single_map(self, java_map: jpype.JObject,
                                 conversion_fn: Callable = lambda x: x):
        return {key: conversion_fn(value) for key, value in java_map.entrySet()}

    def setPlacement(self, action: int) -> None:
        self.instance.setPlacement(action)
