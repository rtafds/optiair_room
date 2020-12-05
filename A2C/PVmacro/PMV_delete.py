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

# get color transfer function/color map for 'PMV'
pMVLUT = GetColorTransferFunction('PMV')
pMVLUT.RGBPoints = [-2.4178788661956787, 0.231373, 0.298039, 0.752941, -0.2670940160751343, 0.865003, 0.865003, 0.865003, 1.8836908340454102, 0.705882, 0.0156863, 0.14902]
pMVLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
temporalInterpolator1Display.Representation = 'Surface'
temporalInterpolator1Display.ColorArrayName = ['POINTS', 'PMV']
temporalInterpolator1Display.LookupTable = pMVLUT
temporalInterpolator1Display.OSPRayScaleArray = 'PMV'
temporalInterpolator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
temporalInterpolator1Display.SelectOrientationVectors = 'None'
temporalInterpolator1Display.ScaleFactor = 0.8
temporalInterpolator1Display.SelectScaleArray = 'None'
temporalInterpolator1Display.GlyphType = 'Arrow'
temporalInterpolator1Display.GlyphTableIndexArray = 'None'
temporalInterpolator1Display.GaussianRadius = 0.04
temporalInterpolator1Display.SetScaleArray = ['POINTS', 'PMV']
temporalInterpolator1Display.ScaleTransferFunction = 'PiecewiseFunction'
temporalInterpolator1Display.OpacityArray = ['POINTS', 'PMV']
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

# get opacity transfer function/opacity map for 'PMV'
pMVPWF = GetOpacityTransferFunction('PMV')
pMVPWF.Points = [-2.4178788661956787, 0.0, 0.5, 0.0, 1.8836908340454102, 1.0, 0.5, 0.0]
pMVPWF.ScalarRangeInitialized = 1

# set active source
SetActiveSource(temporalInterpolator1)

# find source
case2OpenFOAM = FindSource('case2.OpenFOAM')

# set active source
SetActiveSource(case2OpenFOAM)

# hide data in view
Hide(temporalInterpolator1, renderView1)

# show data in view
case2OpenFOAMDisplay = Show(case2OpenFOAM, renderView1)

# trace defaults for the display properties.
case2OpenFOAMDisplay.Representation = 'Surface'
case2OpenFOAMDisplay.ColorArrayName = ['POINTS', 'PMV']
case2OpenFOAMDisplay.LookupTable = pMVLUT
case2OpenFOAMDisplay.OSPRayScaleArray = 'T'
case2OpenFOAMDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
case2OpenFOAMDisplay.SelectOrientationVectors = 'None'
case2OpenFOAMDisplay.ScaleFactor = 0.8
case2OpenFOAMDisplay.SelectScaleArray = 'None'
case2OpenFOAMDisplay.GlyphType = 'Arrow'
case2OpenFOAMDisplay.GlyphTableIndexArray = 'None'
case2OpenFOAMDisplay.GaussianRadius = 0.04
case2OpenFOAMDisplay.SetScaleArray = ['POINTS', 'T']
case2OpenFOAMDisplay.ScaleTransferFunction = 'PiecewiseFunction'
case2OpenFOAMDisplay.OpacityArray = ['POINTS', 'T']
case2OpenFOAMDisplay.OpacityTransferFunction = 'PiecewiseFunction'
case2OpenFOAMDisplay.DataAxesGrid = 'GridAxesRepresentation'
case2OpenFOAMDisplay.SelectionCellLabelFontFile = ''
case2OpenFOAMDisplay.SelectionPointLabelFontFile = ''
case2OpenFOAMDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
case2OpenFOAMDisplay.DataAxesGrid.XTitleFontFile = ''
case2OpenFOAMDisplay.DataAxesGrid.YTitleFontFile = ''
case2OpenFOAMDisplay.DataAxesGrid.ZTitleFontFile = ''
case2OpenFOAMDisplay.DataAxesGrid.XLabelFontFile = ''
case2OpenFOAMDisplay.DataAxesGrid.YLabelFontFile = ''
case2OpenFOAMDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
case2OpenFOAMDisplay.PolarAxes.PolarAxisTitleFontFile = ''
case2OpenFOAMDisplay.PolarAxes.PolarAxisLabelFontFile = ''
case2OpenFOAMDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
case2OpenFOAMDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show color bar/color legend
case2OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

# destroy temporalInterpolator1
Delete(temporalInterpolator1)
del temporalInterpolator1

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).