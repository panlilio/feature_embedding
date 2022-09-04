"""PyQT-based visualization of the latent space representation."""
import sys
from argparse import ArgumentParser

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5 import QtCore, QtWidgets


def load_image(image):
    # TODO Read the image from file
    return image


class Results:
    """
    Read feature embedding results directory and serve samples and latents.
    """
    # TODO initialize from results file, currently a dummy
    def __init__(self, dirname):
        # TODO Get the UMAP/TSNE of your embedding
        self.x, self.y = np.random.randn(2, 100)
        # TODO Get a list of image names from your raw file names
        self.raw = np.random.randn(100, 5, 5)
        # TODO Get a list of image names from your predicted file names
        self.predicted = np.random.randn(100, 5, 5)

    def __getitem__(self, idx):
        raw = load_image(self.raw[idx])
        predicted = load_image(self.predicted[idx])
        return raw, predicted

    def __len__(self):
        return len(self.x)


class Annotation(QtWidgets.QWidget):
    def __init__(self, cmap, parent=None):
        """Plot a clickable colormap for selecting labels.

        Assumes a continuous colormap.
        """
        super().__init__(parent=parent)
        self.parent = parent
        self.cmap = matplotlib.cm.get_cmap(cmap)
        # Create figure and adjust figure height to number of colormaps
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 1))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect('button_press_event', self.on_click)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # Plot the colormap gradient for clicking
        self.gradient = np.linspace(0, 1, 256)[None, :]
        # self.gradient = np.vstack((gradient, gradient))
        self.ax.imshow(self.gradient, aspect=0.1*256, cmap=cmap)
        # Turn off *all* ticks & spines, not just the ones with colormaps.
        self.ax.set_axis_off()
        # TODO Remove the whitespace around the canvas

    def on_click(self, event):
        cval = self.gradient[0, int(event.xdata)]
        self.parent.on_color(self.cmap(cval))


class ScatterPlot(QtWidgets.QWidget):
    """Clickable scatterplot

    When a point in the plot is clicked, its index is returned to the parent.
    """
    def __init__(self, results, labels=None, parent=None):
        super().__init__(parent=parent)
        self.figure, self.ax = plt.subplot_mosaic("""
                                                  ssssrr
                                                  ssssrr
                                                  sssspp
                                                  sssspp
                                                  """)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect('pick_event', self.on_pick)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # Scatter
        self.results = results
        self.data = self.ax['s'].scatter(self.results.x, self.results.y,
                                         facecolors=["k"]*len(self.results.x),
                                         picker=5)

    def on_pick(self, event):
        """Show images corresponding to chosen point."""
        self.current = event.ind[0]
        # print(f"Drawing sample {current}")
        raw, pred = self.results[self.current]
        self.ax['r'].imshow(np.moveaxis(raw, 0, -1))
        self.ax['p'].imshow(np.moveaxis(pred, 0, -1))
        self.canvas.draw()

    def recolor(self, rgba):
        """
        rgba : Tuple
        """
        self.data._facecolors[self.current, :] = rgba
        self.canvas.draw()


class Visualizer(QtWidgets.QMainWindow):
    """Visualizing autoencoder latent space"""
    def __init__(self, dirname, cmap="inferno"):
        super().__init__()
        results = Results(dirname)
        self.scatter = ScatterPlot(results, parent=self)
        self.annotate = Annotation(cmap, parent=self)
        # Create layout
        multiwidget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.annotate)
        layout.addWidget(self.scatter)
        multiwidget.setLayout(layout)
        self.setCentralWidget(multiwidget)
        self.center()

    def center(self):
        # Get the resolution of the screen
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        # Get the size of widget
        size = self.geometry()
        self.move((screen.width() - size.width())/2,
                  (screen.height() - size.height())/2)

    def on_color(self, color):
        """Pass coloring from the annotation to the Scatter Plot"""
        self.scatter.recolor(color)


if __name__ == '__main__':
    parser = ArgumentParser("Visualize and Annotate AE latent space.")
    parser.add_argument("dirname", default=None, help="Results directory")
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    b = Visualizer(args.dirname)
    b.show()
    app.exec_()
