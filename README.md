# HydraulicStructures-DataAnalyst

pyinstaller --onedir --console --clean --exclude-module PyQt6 --exclude-module PySide6 --hidden-import=data_ouvrageHydraulique --hidden-import=numpy --hidden-import=scipy --collect-all numpy --collect-all scipy --add-data "*.xlsx;." interface.py

cd *\dist\interface
copy "C:\Users\*\anaconda3\Library\bin\mkl*.dll" .
copy "C:\Users\*\anaconda3\Library\bin\libiomp5md.dll" .
copy "C:\Users\*\anaconda3\Library\bin\libmmd.dll" .
copy "C:\Users\*\anaconda3\Library\bin\svml_dispmd.dll" .

#Others
pyinstaller --onefile --windowed --exclude-module PyQt6 --exclude-module PySide6 interface.py

pyinstaller --onefile --console --debug=all --clean --exclude-module PyQt6 --exclude-module PySide6 interface.py

pyinstaller --onedir --windowed  --exclude-module PyQt6 --exclude-module PySide6 interface.py

pyinstaller --onedir --console --clean interface.py dist\interface\interface.exe

pyinstaller --onedir --console --clean --exclude-module PyQt6 --exclude-module PySide6 interface.py

pyinstaller --onedir --console --clean --exclude-module PyQt6 --exclude-module PySide6 --hidden-import=data_ouvrageHydraulique --add-data "*.xlsx;." interface.py
