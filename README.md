# HydraulicStructures-DataAnalyst

pyinstaller --onedir --console --clean --exclude-module PyQt6 --exclude-module PySide6 --hidden-import=data_ouvrageHydraulique --hidden-import=numpy --hidden-import=scipy --collect-all numpy --collect-all scipy --add-data "*.xlsx;." interface.py

cd *\dist\interface
copy "C:\Users\*\anaconda3\Library\bin\mkl*.dll" .
copy "C:\Users\*\anaconda3\Library\bin\libiomp5md.dll" .
copy "C:\Users\*\anaconda3\Library\bin\libmmd.dll" .
copy "C:\Users\*\anaconda3\Library\bin\svml_dispmd.dll" .
