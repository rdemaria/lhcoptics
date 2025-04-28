LHCRepo
=======

The ``LHCRepo`` class is responsible for managing the LHC optics repository data. It provides functionality to:

- Access fill data and beam process information
- Retrieve optics configurations for specific fills
- Match beam processes to fill data
- Extract relevant optics parameters

Usage:
------

.. code-block:: python

    from lhcoptics import LHCRepo
    
    # Initialize the repository
    repo = LHCRepo()
    
    # Get fills with specific beam processes
    fills = repo.get_fills_with_bp(['LHCBEAM1/QH_TRIM_FIDEL', 'LHCBEAM1/QH_TRIM_INT'])
    
    # Access fill data
    fill_data = repo.get_fill(7920)

API Reference:
--------------

.. autoclass:: lhcoptics.repo.LHCRepo
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex: