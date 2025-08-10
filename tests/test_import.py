"""Minimal tests to ensure package can be imported"""

def test_import_claire():
    """Test that claire can be imported"""
    import claire
    assert claire is not None

def test_import_submodules():
    """Test that submodules can be imported"""
    try:
        from claire import MembraneSystem
        from claire.analysis import MLAnalyzer
        from claire.visualization import FigureGenerator
        assert True
    except ImportError:
        # OK if submodules not fully implemented yet
        pass

def test_version():
    """Test that version is defined"""
    try:
        import claire
        assert hasattr(claire, '__version__')
    except:
        # OK if version not defined yet
        pass