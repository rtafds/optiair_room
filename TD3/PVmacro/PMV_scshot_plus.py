# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# get active source.
case3OpenFOAM = GetActiveSource()

# Properties modified on case3OpenFOAM
case3OpenFOAM.VolumeFields = ['PMV', 'PPD', 'T', 'U', 'p', 'p_rgh']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [979, 688]

# update the view to ensure updated data information
renderView1.Update()

# get display properties
case3OpenFOAMDisplay = GetDisplayProperties(case3OpenFOAM, view=renderView1)

# get color transfer function/color map for 'vtkBlockColors'
vtkBlockColorsLUT = GetColorTransferFunction('vtkBlockColors')
vtkBlockColorsLUT.InterpretValuesAsCategories = 1
vtkBlockColorsLUT.AnnotationsInitialized = 1
vtkBlockColorsLUT.Annotations = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5', '5', '6', '6', '7', '7', '8', '8', '9', '9', '10', '10', '11', '11']
vtkBlockColorsLUT.ActiveAnnotatedValues = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
vtkBlockColorsLUT.IndexedColors = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.6299992370489051, 0.6299992370489051, 1.0, 0.6699931334401464, 0.5000076295109483, 0.3300068665598535, 1.0, 0.5000076295109483, 0.7499961852445258, 0.5300068665598535, 0.3499961852445258, 0.7000076295109483, 1.0, 0.7499961852445258, 0.5000076295109483]
vtkBlockColorsLUT.IndexedOpacities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# set scalar coloring
ColorBy(case3OpenFOAMDisplay, ('POINTS', 'PMV'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(vtkBlockColorsLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
case3OpenFOAMDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
case3OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'PMV'
pMVLUT = GetColorTransferFunction('PMV')
pMVLUT.RGBPoints = [2.89599347114563, 0.231373, 0.298039, 0.752941, 2.89623761177063, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]
pMVLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'PMV'
pMVPWF = GetOpacityTransferFunction('PMV')
pMVPWF.Points = [2.89599347114563, 0.0, 0.5, 0.0, 2.89648175239563, 1.0, 0.5, 0.0]
pMVPWF.ScalarRangeInitialized = 1

# get animation scene
animationScene1 = GetAnimationScene()

animationScene1.Play()

# set scalar coloring
ColorBy(case3OpenFOAMDisplay, ('POINTS', 'p'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pMVLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
case3OpenFOAMDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
case3OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')
pLUT.RGBPoints = [101298.0, 0.231373, 0.298039, 0.752941, 101311.5, 0.865003, 0.865003, 0.865003, 101325.0, 0.705882, 0.0156863, 0.14902]
pLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')
pPWF.Points = [101298.0, 0.0, 0.5, 0.0, 101325.0, 1.0, 0.5, 0.0]
pPWF.ScalarRangeInitialized = 1

animationScene1.Play()

# set scalar coloring
ColorBy(case3OpenFOAMDisplay, ('POINTS', 'PPD'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
case3OpenFOAMDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
case3OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'PPD'
pPDLUT = GetColorTransferFunction('PPD')
pPDLUT.RGBPoints = [7.349340915679932, 0.231373, 0.298039, 0.752941, 49.429728746414185, 0.865003, 0.865003, 0.865003, 91.51011657714844, 0.705882, 0.0156863, 0.14902]
pPDLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'PPD'
pPDPWF = GetOpacityTransferFunction('PPD')
pPDPWF.Points = [7.349340915679932, 0.0, 0.5, 0.0, 91.51011657714844, 1.0, 0.5, 0.0]
pPDPWF.ScalarRangeInitialized = 1

# set scalar coloring
ColorBy(case3OpenFOAMDisplay, ('POINTS', 'PMV'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pPDLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
case3OpenFOAMDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
case3OpenFOAMDisplay.SetScalarBarVisibility(renderView1, True)

animationScene1.GoToFirst()

# create a new 'Temporal Interpolator'
temporalInterpolator1 = TemporalInterpolator(Input=case3OpenFOAM)
temporalInterpolator1.DiscreteTimeStepInterval = 0.5

# Properties modified on temporalInterpolator1
temporalInterpolator1.DiscreteTimeStepInterval = 10.0

# show data in view
temporalInterpolator1Display = Show(temporalInterpolator1, renderView1)

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

# hide data in view
Hide(case3OpenFOAM, renderView1)

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
SaveScreenshot('./PMV_first.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_500.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_1000.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_1500.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_2000.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_2500.png', renderView1, ImageResolution=[979, 688])

animationScene1.GoToLast()

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_last.png', renderView1, ImageResolution=[979, 688])

# set active source
SetActiveSource(temporalInterpolator1)

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, 0.23994842171669006, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, 0.039455346763134, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -0.1109144538640976, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -0.46177732944488525, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -0.9964255094528198, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -1.5143659114837646, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -2.182676076889038, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -2.2996304035186768, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -2.4165847301483154, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.705882, 0.0156863, 0.14902]

# Properties modified on pMVLUT
pMVLUT.RGBPoints = [-2.4165847301483154, 0.231373, 0.298039, 0.752941, -2.4165847301483154, 0.865003, 0.865003, 0.865003, 2.89648175239563, 0.11764705882352941, 0.11764705882352941, 0.11764705882352941]

animationScene1.GoToFirst()

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_first_gray.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_500_gray.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_1000_gray.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_1500_gray.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_2000_gray.png', renderView1, ImageResolution=[979, 688])

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_2500_gray.png', renderView1, ImageResolution=[979, 688])

animationScene1.GoToLast()

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

# save screenshot
SaveScreenshot('./PMV_last_gray.png', renderView1, ImageResolution=[979, 688])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [4.0, 1.2000000476837158, 16.18645520483465]
renderView1.CameraFocalPoint = [4.0, 1.2000000476837158, 0.05000000074505806]
renderView1.CameraParallelScale = 4.17642192726207

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).