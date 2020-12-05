# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
temporalInterpolator1 = FindSource('TemporalInterpolator1')

# set active source
SetActiveSource(temporalInterpolator1)

# find source
annotateTimeFilter1 = FindSource('AnnotateTimeFilter1')

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [979, 688]

# hide data in view
Hide(annotateTimeFilter1, renderView1)

# show data in view
temporalInterpolator1Display = Show(temporalInterpolator1, renderView1)

# get color transfer function/color map for 'T'
tLUT = GetColorTransferFunction('T')
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 298.5749969482422, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]
tLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
temporalInterpolator1Display.Representation = 'Surface'
temporalInterpolator1Display.ColorArrayName = ['POINTS', 'T']
temporalInterpolator1Display.LookupTable = tLUT
temporalInterpolator1Display.OSPRayScaleArray = 'T'
temporalInterpolator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
temporalInterpolator1Display.SelectOrientationVectors = 'None'
temporalInterpolator1Display.ScaleFactor = 0.8
temporalInterpolator1Display.SelectScaleArray = 'None'
temporalInterpolator1Display.GlyphType = 'Arrow'
temporalInterpolator1Display.GlyphTableIndexArray = 'None'
temporalInterpolator1Display.GaussianRadius = 0.04
temporalInterpolator1Display.SetScaleArray = ['POINTS', 'T']
temporalInterpolator1Display.ScaleTransferFunction = 'PiecewiseFunction'
temporalInterpolator1Display.OpacityArray = ['POINTS', 'T']
temporalInterpolator1Display.OpacityTransferFunction = 'PiecewiseFunction'
temporalInterpolator1Display.DataAxesGrid = 'GridAxesRepresentation'
temporalInterpolator1Display.SelectionCellLabelFontFile = ''
temporalInterpolator1Display.SelectionPointLabelFontFile = ''
temporalInterpolator1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
temporalInterpolator1Display.DataAxesGrid.XTitleFontFile = ''
temporalInterpolator1Display.DataAxesGrid.YTitleFontFile = ''
temporalInterpolator1Display.DataAxesGrid.ZTitleFontFile = ''
temporalInterpolator1Display.DataAxesGrid.XLabelFontFile = ''
temporalInterpolator1Display.DataAxesGrid.YLabelFontFile = ''
temporalInterpolator1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
temporalInterpolator1Display.PolarAxes.PolarAxisTitleFontFile = ''
temporalInterpolator1Display.PolarAxes.PolarAxisLabelFontFile = ''
temporalInterpolator1Display.PolarAxes.LastRadialAxisTextFontFile = ''
temporalInterpolator1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
temporalInterpolator1Display.SetScalarBarVisibility(renderView1, True)

# destroy annotateTimeFilter1
Delete(annotateTimeFilter1)
del annotateTimeFilter1

# get opacity transfer function/opacity map for 'T'
tPWF = GetOpacityTransferFunction('T')
tPWF.Points = [291.1499938964844, 0.0, 0.5, 0.0, 306.0, 1.0, 0.5, 0.0]
tPWF.ScalarRangeInitialized = 1

# set active source
SetActiveSource(temporalInterpolator1)

# find source
glyph1 = FindSource('Glyph1')

# set active source
SetActiveSource(glyph1)

# hide data in view
Hide(temporalInterpolator1, renderView1)

# show data in view
glyph1Display = Show(glyph1, renderView1)

# trace defaults for the display properties.
glyph1Display.Representation = 'Surface'
glyph1Display.ColorArrayName = ['POINTS', 'T']
glyph1Display.LookupTable = tLUT
glyph1Display.OSPRayScaleArray = 'T'
glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
glyph1Display.SelectOrientationVectors = 'None'
glyph1Display.ScaleFactor = 0.8
glyph1Display.SelectScaleArray = 'None'
glyph1Display.GlyphType = 'Arrow'
glyph1Display.GlyphTableIndexArray = 'None'
glyph1Display.GaussianRadius = 0.04
glyph1Display.SetScaleArray = ['POINTS', 'T']
glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
glyph1Display.OpacityArray = ['POINTS', 'T']
glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
glyph1Display.SelectionCellLabelFontFile = ''
glyph1Display.SelectionPointLabelFontFile = ''
glyph1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
glyph1Display.DataAxesGrid.XTitleFontFile = ''
glyph1Display.DataAxesGrid.YTitleFontFile = ''
glyph1Display.DataAxesGrid.ZTitleFontFile = ''
glyph1Display.DataAxesGrid.XLabelFontFile = ''
glyph1Display.DataAxesGrid.YLabelFontFile = ''
glyph1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
glyph1Display.PolarAxes.PolarAxisTitleFontFile = ''
glyph1Display.PolarAxes.PolarAxisLabelFontFile = ''
glyph1Display.PolarAxes.LastRadialAxisTextFontFile = ''
glyph1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
glyph1Display.SetScalarBarVisibility(renderView1, True)

# destroy temporalInterpolator1
Delete(temporalInterpolator1)
del temporalInterpolator1

# find source
case4OpenFOAM = FindSource('case4.OpenFOAM')

# set active source
SetActiveSource(case4OpenFOAM)

# hide data in view
Hide(glyph1, renderView1)

# show data in view
case4OpenFOAMDisplay = Show(case4OpenFOAM, renderView1)

# trace defaults for the display properties.
case4OpenFOAMDisplay.Representation = 'Surface'
case4OpenFOAMDisplay.ColorArrayName = ['POINTS', 'T']
case4OpenFOAMDisplay.LookupTable = tLUT
case4OpenFOAMDisplay.OSPRayScaleArray = 'T'
case4OpenFOAMDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
case4OpenFOAMDisplay.SelectOrientationVectors = 'None'
case4OpenFOAMDisplay.ScaleFactor = 0.8
case4OpenFOAMDisplay.SelectScaleArray = 'None'
case4OpenFOAMDisplay.GlyphType = 'Arrow'
case4OpenFOAMDisplay.GlyphTableIndexArray = 'None'
case4OpenFOAMDisplay.GaussianRadius = 0.04
case4OpenFOAMDisplay.SetScaleArray = ['POINTS', 'T']
case4OpenFOAMDisplay.ScaleTransferFunction = 'PiecewiseFunction'
case4OpenFOAMDisplay.OpacityArray = ['POINTS', 'T']
case4OpenFOAMDisplay.OpacityTransferFunction = 'PiecewiseFunction'
case4OpenFOAMDisplay.DataAxesGrid = 'GridAxesRepresentation'
case4OpenFOAMDisplay.SelectionCellLabelFontFile = ''
case4OpenFOAMDisplay.SelectionPointLabelFontFile = ''
case4OpenFOAMDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
case4OpenFOAMDisplay.DataAxesGrid.XTitleFontFile = ''
case4OpenFOAMDisplay.DataAxesGrid.YTitleFontFile = ''
case4OpenFOAMDisplay.DataAxesGrid.ZTitleFontFile = ''
case4OpenFOAMDisplay.DataAxesGrid.XLabelFontFile = ''
case4OpenFOAMDisplay.DataAxesGrid.YLabelFontFile = ''
case4OpenFOAMDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
case4OpenFOAMDisplay.PolarAxes.PolarAxisTitleFontFile = ''
case4OpenFOAMDisplay.PolarAxes.PolarAxisLabelFontFile = ''
case4OpenFOAMDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
case4OpenFOAMDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
case4OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

# destroy glyph1
Delete(glyph1)
del glyph1