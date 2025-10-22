"""
error_handling.py - Comprehensive error handling utilities for the simulation

Part of the Human Society Simulation project.

Provides centralized error handling, logging, and recovery mechanisms
for robust simulation execution.
"""

import os
import sys
import logging
import traceback
from typing import Any, Optional, Callable, Union
from contextlib import contextmanager
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ───────────────────────── Error Classes ─────────────────────────

class SimulationError(Exception):
    """Base exception for simulation-specific errors."""
    pass

class ConfigurationError(SimulationError):
    """Raised when configuration parameters are invalid."""
    pass

class ResourceError(SimulationError):
    """Raised when resource operations fail."""
    pass

class AgentError(SimulationError):
    """Raised when agent operations fail."""
    pass

class ValidationError(SimulationError):
    """Raised when data validation fails."""
    pass

# ───────────────────────── Error Handling Utilities ─────────────────────────

def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """
    Safely execute a function and return success status and result.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (success: bool, result: Any)
        
    Example:
        >>> success, result = safe_execute(risky_function, param1, param2)
        >>> if success:
        ...     print(f"Result: {result}")
        ... else:
        ...     print("Function failed safely")
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Function {func.__name__} failed: {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        return False, e

@contextmanager
def error_context(operation: str, reraise: bool = True):
    """
    Context manager for error handling with logging.
    
    Args:
        operation: Description of the operation being performed
        reraise: Whether to reraise exceptions (default True)
        
    Example:
        >>> with error_context("Loading simulation data"):
        ...     data = load_simulation_data()
    """
    try:
        logger.info(f"Starting: {operation}")
        yield
        logger.info(f"Completed: {operation}")
    except Exception as e:
        logger.error(f"Failed: {operation} - {e}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        if reraise:
            raise

def validate_file_path(file_path: str, must_exist: bool = True) -> str:
    """
    Validate file path and return normalized path.
    
    Args:
        file_path: Path to validate
        must_exist: Whether file must exist (default True)
        
    Returns:
        Normalized file path
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True
        ValueError: If path is invalid
        
    Example:
        >>> path = validate_file_path("data/simulation.csv")
        >>> print(f"Valid path: {path}")
    """
    if not isinstance(file_path, str):
        raise TypeError(f"File path must be a string, got {type(file_path)}")
    
    if not file_path.strip():
        raise ValueError("File path cannot be empty")
    
    # Normalize path
    normalized_path = os.path.normpath(file_path)
    
    if must_exist and not os.path.exists(normalized_path):
        raise FileNotFoundError(f"File not found: {normalized_path}")
    
    return normalized_path

def validate_bounds(value: Union[int, float], min_val: Union[int, float], 
                   max_val: Union[int, float], name: str = "value") -> Union[int, float]:
    """
    Validate that a value is within specified bounds.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValueError: If value is outside bounds
        
    Example:
        >>> age = validate_bounds(25, 0, 120, "age")
        >>> print(f"Valid age: {age}")
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value)}")
    
    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise TypeError("min_val and max_val must be numbers")
    
    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    return value

def validate_positive(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValueError: If value is not positive
        
    Example:
        >>> count = validate_positive(42, "count")
        >>> print(f"Valid count: {count}")
    """
    return validate_bounds(value, 0.001, float('inf'), name)

def validate_integer(value: Any, name: str = "value") -> int:
    """
    Validate that a value is an integer.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        The validated integer
        
    Raises:
        TypeError: If value is not an integer
        
    Example:
        >>> id_num = validate_integer(123, "ID")
        >>> print(f"Valid ID: {id_num}")
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value)}")
    
    return value

def validate_choice(value: Any, choices: list, name: str = "value") -> Any:
    """
    Validate that a value is one of the allowed choices.
    
    Args:
        value: Value to validate
        choices: List of allowed values
        name: Name of the value for error messages
        
    Returns:
        The validated value
        
    Raises:
        ValueError: If value is not in choices
        
    Example:
        >>> sex = validate_choice('M', ['M', 'F'], "sex")
        >>> print(f"Valid sex: {sex}")
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")
    
    return value

# ───────────────────────── Recovery Mechanisms ─────────────────────────

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts (default 3)
        delay: Delay between attempts in seconds (default 1.0)
        exceptions: Tuple of exception types to catch (default all)
        
    Returns:
        Decorated function
        
    Example:
        >>> @retry_on_failure(max_attempts=5, delay=0.5)
        ... def unreliable_function():
        ...     # Function that might fail
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator

def fallback_value(default_value: Any, exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to return a fallback value on failure.
    
    Args:
        default_value: Value to return on failure
        exceptions: Tuple of exception types to catch (default all)
        
    Returns:
        Decorated function
        
    Example:
        >>> @fallback_value(default_value=0)
        ... def risky_calculation():
        ...     # Function that might fail
        ...     return complex_calculation()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(f"Function {func.__name__} failed, using fallback value: {e}")
                return default_value
        
        return wrapper
    return decorator

# ───────────────────────── Simulation-Specific Error Handling ─────────────────────────

def validate_simulation_state(humans: list, houses: list, resources: Any) -> None:
    """
    Validate the current state of the simulation.
    
    Args:
        humans: List of Human agents
        houses: List of House objects
        resources: Resource array
        
    Raises:
        ValidationError: If simulation state is invalid
        
    Example:
        >>> validate_simulation_state(humans, houses, resources)
        >>> print("Simulation state is valid")
    """
    # Validate humans
    if not isinstance(humans, list):
        raise ValidationError(f"humans must be a list, got {type(humans)}")
    
    for i, human in enumerate(humans):
        if not hasattr(human, 'id'):
            raise ValidationError(f"Human {i} missing 'id' attribute")
        
        if not hasattr(human, 'alive'):
            raise ValidationError(f"Human {i} missing 'alive' attribute")
        
        if not hasattr(human, 'x') or not hasattr(human, 'y'):
            raise ValidationError(f"Human {i} missing position attributes")
    
    # Validate houses
    if not isinstance(houses, list):
        raise ValidationError(f"houses must be a list, got {type(houses)}")
    
    for i, house in enumerate(houses):
        if not hasattr(house, 'x') or not hasattr(house, 'y'):
            raise ValidationError(f"House {i} missing position attributes")
    
    # Validate resources
    if resources is None:
        raise ValidationError("resources cannot be None")
    
    if not hasattr(resources, 'shape'):
        raise ValidationError("resources must have shape attribute")
    
    logger.info(f"Simulation state validated: {len(humans)} humans, {len(houses)} houses")

def handle_simulation_error(error: Exception, context: str = "") -> None:
    """
    Handle simulation errors with appropriate logging and recovery.
    
    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     handle_simulation_error(e, "during food spawning")
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    logger.error(f"Simulation error {context}: {error_type}: {error_msg}")
    logger.debug(f"Full traceback:\n{traceback.format_exc()}")
    
    # Specific handling for different error types
    if isinstance(error, ValidationError):
        logger.warning("Validation error - simulation state may be corrupted")
    elif isinstance(error, ResourceError):
        logger.warning("Resource error - attempting recovery")
    elif isinstance(error, AgentError):
        logger.warning("Agent error - individual agent may be affected")
    elif isinstance(error, ConfigurationError):
        logger.critical("Configuration error - simulation cannot continue")
        raise  # Reraise configuration errors as they're fatal
    
    # For non-fatal errors, continue simulation with warnings
    warnings.warn(f"Simulation error {context}: {error_msg}", UserWarning)

# ───────────────────────── Public API ─────────────────────────

__all__ = [
    'SimulationError', 'ConfigurationError', 'ResourceError', 'AgentError', 'ValidationError',
    'safe_execute', 'error_context', 'validate_file_path', 'validate_bounds', 
    'validate_positive', 'validate_integer', 'validate_choice',
    'retry_on_failure', 'fallback_value', 'validate_simulation_state', 'handle_simulation_error'
]
