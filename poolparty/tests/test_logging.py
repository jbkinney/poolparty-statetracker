"""Tests for logging configuration."""
import logging
import poolparty as pp


def test_logging_default_nullhandler():
    """Test that logging uses NullHandler by default (no output)."""
    # Clear any existing loggers
    for logger_name in ("poolparty", "statetracker"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
    
    # Create a pool without configuring logging - should not produce output
    pp.init()
    pool = pp.from_seq("ACGT")
    # If NullHandler is working, no errors or warnings about missing handlers


def test_logging_configuration(caplog):
    """Test that logging can be enabled and produces output."""
    # Configure logging at DEBUG level
    pp.init(log_level="DEBUG")
    
    # Capture logs at DEBUG level
    with caplog.at_level(logging.DEBUG):
        pool = pp.from_seq("ACGT")
        result = pool.mutagenize(num_mutations=1, mode='random', num_states=2)
        df = result.generate_library(num_cycles=1, seed=42)
    
    # Verify we captured some log messages
    assert len(caplog.records) > 0, "No log records captured"
    
    # Verify we have poolparty logs
    poolparty_logs = [r for r in caplog.records if "poolparty" in r.name]
    assert len(poolparty_logs) > 0, "No poolparty log records found"
    
    # Verify we have statetracker logs
    statetracker_logs = [r for r in caplog.records if "statetracker" in r.name]
    assert len(statetracker_logs) > 0, "No statetracker log records found"


def test_logging_info_level(caplog):
    """Test INFO level logging."""
    # Configure at INFO level
    pp.init(log_level="INFO")
    
    with caplog.at_level(logging.INFO):
        pool = pp.from_seq("ACGT")
        df = pool.generate_library(num_cycles=1)
    
    # Should have INFO messages but fewer than DEBUG
    assert len(caplog.records) > 0
    
    # Check for specific INFO messages
    info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
    assert any("library generation" in msg.lower() for msg in info_messages)


def test_logging_levels():
    """Test that different log levels work correctly."""
    # Test DEBUG
    pp.init(log_level="DEBUG")
    logger = logging.getLogger("poolparty")
    assert logger.level == logging.DEBUG
    
    # Test INFO
    pp.configure_logging(level="INFO")
    assert logger.level == logging.INFO
    
    # Test WARNING
    pp.configure_logging(level="WARNING")
    assert logger.level == logging.WARNING
    
    # Test ERROR
    pp.configure_logging(level="ERROR")
    assert logger.level == logging.ERROR


def test_configure_logging_function():
    """Test the configure_logging function directly."""
    # Should accept level as string (case insensitive)
    pp.configure_logging(level="debug")
    logger = logging.getLogger("poolparty")
    assert logger.level == logging.DEBUG
    
    pp.configure_logging(level="INFO")
    assert logger.level == logging.INFO
    
    # Both poolparty and statetracker should be configured
    st_logger = logging.getLogger("statetracker")
    assert st_logger.level == logging.INFO
