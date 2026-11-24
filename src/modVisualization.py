"""Abaqus post-processing for the FFMcracking examples."""

import math
import os
import shutil

from abaqus import session
from abaqusConstants import (
    COMPONENT, CONTOURS_ON_DEF, CONTOURS_ON_UNDEF, FILLED_CIRCLE,
    FILLED_DIAMOND, INTEGRATION_POINT, NONE, OFF, ON, PNG, SHADED, SYMBOL,
    UNIFORM,
)
# These modules register the ODB and XY plotting methods on the Abaqus session.
import visualization
import xyPlot


class CrackResults(object):
    """Keep and display the cracked ODB for the lowest factor recorded so far."""

    def __init__(self, model):
        job_name = os.path.splitext(model.inpFileName)[0] + '-EC-with'
        self.source = os.path.join(model.outRoute, job_name + '.odb')
        self.directory = os.path.join(model.outRoute, 'visualization')
        self.odb_path = os.path.join(self.directory, 'crack.odb').replace(os.sep, '/')
        self.step = model.nameStep
        self.best_factor = None
        self.snapshot_ready = False

    def save_if_best(self, factor):
        # Match the optimizer's strict comparison so ties keep the first result.
        if self.best_factor is not None and not factor < self.best_factor:
            return
        self.best_factor = factor
        self.snapshot_ready = False

        pending_path = self.odb_path + '.tmp'
        try:
            if not os.path.isdir(self.directory):
                os.makedirs(self.directory)
            shutil.copyfile(self.source, pending_path)
            if self.odb_path in session.odbs:
                session.odbs[self.odb_path].close()
            if os.path.exists(self.odb_path):
                os.remove(self.odb_path)
            os.rename(pending_path, self.odb_path)
            self.snapshot_ready = True
        except Exception as error:
            print('FFM visualization: could not save the cracked ODB: {}'.format(error))
        finally:
            if os.path.isfile(pending_path):
                try:
                    os.remove(pending_path)
                except OSError as error:
                    print('FFM visualization: could not remove {}: {}'.format(pending_path, error))

    def show(self, lengths=None, factors=None):
        """Export the figures and leave the cracked contour visible in CAE."""
        viewport = session.viewports['Viewport: 1']
        if lengths is not None:
            try:
                self._show_optimization(viewport, lengths, factors)
            except Exception as error:
                print('FFM visualization: could not show the optimization: {}'.format(error))
        if self.snapshot_ready:
            try:
                self._show_crack(viewport)
            except Exception as error:
                print('FFM visualization: could not show the cracked model: {}'.format(error))

    def _export(self, viewport, filename):
        path = os.path.join(self.directory, filename)
        options = session.printOptions
        previous = (options.vpDecorations, options.vpBackground)
        try:
            options.setValues(vpDecorations=OFF, vpBackground=OFF)
            session.printToFile(fileName=path, format=PNG, canvasObjects=(viewport,))
        finally:
            options.setValues(vpDecorations=previous[0], vpBackground=previous[1])
        print('FFM visualization: saved {}'.format(path))

    def _show_crack(self, viewport):
        odb = session.openOdb(name=self.odb_path, readOnly=True)
        keep_open = False
        try:
            viewport.setValues(displayedObject=odb)
            display = viewport.odbDisplay
            step = odb.steps[self.step]
            display.setFrame(step=list(odb.steps.keys()).index(self.step),
                             frame=len(step.frames) - 1)
            display.commonOptions.setValues(renderStyle=SHADED, visibleEdges=NONE)
            display.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT,
                                       refinement=(COMPONENT, 'S22'))
            display.display.setValues(plotState=(CONTOURS_ON_UNDEF,))
            viewport.viewportAnnotationOptions.setValues(legend=ON, state=ON)
            viewport.view.fitView()
            self._export(viewport, 'crack-s22.png')

            # Magnify the nominal-load displacement so the XFEM opening is visible.
            display.commonOptions.setValues(deformationScaling=UNIFORM, uniformScaleFactor=150.0)
            display.display.setValues(plotState=(CONTOURS_ON_DEF,))
            self._export(viewport, 'crack-opening-s22.png')
            keep_open = session.attachedToGui
        finally:
            # The GUI keeps only the read-only snapshot open, never a solver output.
            if not keep_open:
                odb.close()

    def _show_optimization(self, viewport, lengths, factors):
        points = [(float(length), float(factor)) for length, factor in zip(lengths, factors)]
        finite_points = [(length, factor) for length, factor in points
                         if not math.isinf(factor) and not math.isnan(factor)]
        if len(finite_points) < len(points):
            print('FFM visualization: non-finite factors are omitted from the XY plot.')
        if not finite_points:
            return
        best = min(finite_points, key=lambda point: point[1])
        name = 'FFM crack-length optimization'
        if name not in session.xyPlots:
            session.XYPlot(name=name)
        plot = session.xyPlots[name]
        chart = next(iter(plot.charts.values()))
        observations = _point_curve(tuple(sorted(finite_points)), 'Evaluated lengths',
                                    'Blue', FILLED_CIRCLE, 4.0)
        minimum = _point_curve((best,), 'Lowest observed factor',
                               'Red', FILLED_DIAMOND, 6.0)
        chart.setValues(curvesToPlot=(observations, minimum))
        chart.axes1[0].axisData.setValues(useSystemTitle=False,
                                          title='Candidate crack length (model units)')
        chart.axes2[0].axisData.setValues(useSystemTitle=False,
                                          title='Coupled critical factor')
        viewport.setValues(displayedObject=plot)
        self._export(viewport, 'length-optimization.png')


def _point_curve(points, label, color, marker, size):
    data = xyPlot.XYData(data=points)
    curve = session.Curve(xyData=data)
    curve.setValues(displayTypes=(SYMBOL,), useDefault=OFF, legendLabel=label)
    curve.lineStyle.setValues(show=OFF)
    curve.symbolStyle.setValues(color=color, marker=marker, size=size)
    return curve
