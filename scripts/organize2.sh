# Create the physics directory
mkdir -p physics

# Move physics files (removing the claire: prefix)
mv claire:physics:__init__.py physics/__init__.py
mv claire:physics:calculations.py physics/calculations.py
mv claire:physics:rdf.py physics/rdf.py

# Move analysis files to the analysis directory
mv claire:analysis:diagnostics.py analysis/diagnostics.py
mv claire:analysis:frame_processor.py analysis/frame_processor.py