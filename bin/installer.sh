
pyinstaller  --name="FreeMoCap"  --onefile  --hidden-import pyqtgraph.graphicsItems.ViewBox.axisCtrlTemplate_pyqt6 --hidden-import pyqtgraph.graphicsItems.PlotItem.plotConfigTemplate_pyqt6 --hidden-import pyqtgraph.imageview.ImageViewTemplate_pyqt6  ./src/gui/main/main.py
