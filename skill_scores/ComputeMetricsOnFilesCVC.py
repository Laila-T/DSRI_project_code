import glob
import os

# Modified from Skills2D project

# 1. Open Slicer
# 2. Run this script using the exec command
#    exec(open("C:/path/to/ComputeMetricsOnFilesCVC.py").read())

DATA_DIRECTORY = r"C:\Users\Matthew\Downloads\CVC\2016-TrackingData"
METRICS_DIRECTORY = r"C:\Users\Matthew\Downloads\CVC\Metrics"

def GetFirstTransformNodeContains(name):
  transformNodes = slicer.mrmlScene.GetNodesByClass("vtkMRMLLinearTransformNode")
  transformNodesIterator = vtk.vtkCollectionIterator()
  transformNodesIterator.SetCollection(transformNodes)
  transformNodesIterator.InitTraversal()
  while not transformNodesIterator.IsDoneWithTraversal():
    currentTransformNode = transformNodesIterator.GetCurrentObject()
    if name in currentTransformNode.GetName():
      return currentTransformNode
    transformNodesIterator.GoToNextItem()

  return None


# Create the logic
peLogic = slicer.modules.perkevaluator.logic()

# Iterate over all sessions and load
filepaths = glob.glob(os.path.join(DATA_DIRECTORY, "**", "*.xml"))
for curr_filepath in filepaths:

  slicer.mrmlScene.Clear()

  # Import all of the metric scripts and create the metric instances
  perkEvaluatorNode = slicer.vtkMRMLPerkEvaluatorNode()
  perkEvaluatorNode.SetScene(slicer.mrmlScene)
  slicer.mrmlScene.AddNode(perkEvaluatorNode)

  metricsTableNode = slicer.vtkMRMLTableNode()
  metricsTableNode.SetScene(slicer.mrmlScene)
  slicer.mrmlScene.AddNode(metricsTableNode)

  perkEvaluatorNode.SetMetricsTableID(metricsTableNode.GetID())

  # Create the metrics
  elpasedTimeScript = slicer.util.loadNodeFromFile(os.path.join(METRICS_DIRECTORY, "ElapsedTime.py"), "Python Metric Script", {})
  translationalActionsScript = slicer.util.loadNodeFromFile(os.path.join(METRICS_DIRECTORY, "TranslationalActions.py"), "Python Metric Script", {})

  elaspedTime = peLogic.CreateMetricInstance(elpasedTimeScript)
  perkEvaluatorNode.AddMetricInstanceID(elaspedTime.GetID())

  leftTranslationalActions = peLogic.CreateMetricInstance(translationalActionsScript)
  perkEvaluatorNode.AddMetricInstanceID(leftTranslationalActions.GetID())

  rightTranslationalActions = peLogic.CreateMetricInstance(translationalActionsScript)
  perkEvaluatorNode.AddMetricInstanceID(rightTranslationalActions.GetID())

  # Import the recording file
  try:
    sequenceBrowserNode = slicer.util.loadNodeFromFile(curr_filepath, "Tracked Sequence Browser", {})    
  except:
    print(curr_filepath + " could not be loaded.")
    continue

  # Set the transform roles
  elaspedTime.SetRoleID(GetFirstTransformNodeContains("LeftHandToReference").GetID(), "Any", slicer.vtkMRMLMetricInstanceNode.TransformRole)

  leftTranslationalActions.SetRoleID(GetFirstTransformNodeContains("LeftHandToReference").GetID(), "Any", slicer.vtkMRMLMetricInstanceNode.TransformRole)
  rightTranslationalActions.SetRoleID(GetFirstTransformNodeContains("RightHandToReference").GetID(), "Any", slicer.vtkMRMLMetricInstanceNode.TransformRole)

  # Analyzing...
  perkEvaluatorNode.SetTrackedSequenceBrowserNodeID(sequenceBrowserNode.GetID()) 
  peLogic.ComputeMetrics(perkEvaluatorNode)

  metricsTableNode = perkEvaluatorNode.GetMetricsTableNode()
  slicer.util.saveNode(metricsTableNode, curr_filepath + "_metrics.tsv")

  print("Completed " + curr_filepath + ".")