# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
case2OpenFOAM = GetActiveSource()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [979, 688]

# get display properties
case2OpenFOAMDisplay = GetDisplayProperties(case2OpenFOAM, view=renderView1)

# set scalar coloring
ColorBy(case2OpenFOAMDisplay, ('POINTS', 'T'))

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
vtkBlockColorsLUT.InterpretValuesAsCategories = 1
vtkBlockColorsLUT.AnnotationsInitialized = 1
vtkBlockColorsLUT.Annotations = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11']
vtkBlockColorsLUT.ActiveAnnotatedValues = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
vtkBlockColorsLUT.IndexedColors = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.6299992370489051, 0.6299992370489051, 1.0, 0.6699931334401464, 0.5000076295109483, 0.3300068665598535, 1.0, 0.5000076295109483, 0.7499961852445258, 0.5300068665598535, 0.3499961852445258, 0.7000076295109483, 1.0, 0.7499961852445258, 0.5000076295109483]
vtkBlockColorsLUT.IndexedOpacities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vtkBlockColorsLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
case2OpenFOAMDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
case2OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'T'
tLUT = GetColorTransferFunction('T')
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 298.5749969482422, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]
tLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'T'
tPWF = GetOpacityTransferFunction('T')
tPWF.Points = [291.1499938964844, 0.0, 0.5, 0.0, 306.0, 1.0, 0.5, 0.0]
tPWF.ScalarRangeInitialized = 1

# create a new 'Glyph'
glyph1 = Glyph(Input=case2OpenFOAM,
    GlyphType='Arrow')
glyph1.OrientationArray = ['POINTS', 'No orientation array']
glyph1.ScaleArray = ['POINTS', 'No scale array']
glyph1.ScaleFactor = 0.8
glyph1.GlyphTransform = 'Transform2'

# Properties modified on glyph1
glyph1.ScaleArray = ['POINTS', 'U']

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

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Temporal Interpolator'
temporalInterpolator1 = TemporalInterpolator(Input=glyph1)
temporalInterpolator1.DiscreteTimeStepInterval = 0.5

# Properties modified on temporalInterpolator1
temporalInterpolator1.DiscreteTimeStepInterval = 10.0

# show data in view
temporalInterpolator1Display = Show(temporalInterpolator1, renderView1)

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

# hide data in view
Hide(glyph1, renderView1)

# show color bar/color legend
temporalInterpolator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(Input=temporalInterpolator1)

# show data in view
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1)

# trace defaults for the display properties.
annotateTimeFilter1Display.FontFile = ''

# update the view to ensure updated data information
renderView1.Update()

# Properties modified on annotateTimeFilter1Display
annotateTimeFilter1Display.WindowLocation = 'LowerCenter'
# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./TU_first.png', renderView1, ImageResolution=[979, 688])

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.GoToLast()

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./TU_last.png', renderView1, ImageResolution=[979, 688])

# set active source
SetActiveSource(glyph1)

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 298.2947998046875, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 297.5476379394531, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 296.7070617675781, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 295.9132080078125, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 293.2047119140625, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 291.47686767578125, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 291.1499938964844, 0.865003, 0.865003, 0.865003, 306.0, 0.705882, 0.0156863, 0.14902]

# Properties modified on tLUT
tLUT.RGBPoints = [291.1499938964844, 0.231373, 0.298039, 0.752941, 291.1499938964844, 0.865003, 0.865003, 0.865003, 306.0, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941]

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./TU_last_gray.png', renderView1, ImageResolution=[979, 688])

animationScene1.GoToFirst()

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./TU_first_gray.png', renderView1, ImageResolution=[979, 688])