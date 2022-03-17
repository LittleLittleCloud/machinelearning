using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using Microsoft.Data.Analysis;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.SearchSpace.Tuner;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.Transforms.Text.NgramExtractingEstimator;
using static Microsoft.ML.Transforms.Text.TextNormalizingEstimator;

namespace Microsoft.ML.AutoML.Samples
{
    /// <summary>
    /// titanic experiment using sweepable api and customize search space
    /// </summary>
    internal static class TitanicExperiment
    {
        private static string TrainDataPath = @"C:\\Users\\xiaoyuz\\Desktop\\train.csv";

        public static void Run()
        {
            var context = new MLContext();
            var df = DataFrame.LoadCsv(TrainDataPath);

            var pipeline = context.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair(@"Sex", @"Sex"), new InputOutputColumnPair(@"Embarked", @"Embarked") })
                           .Append(context.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"Pclass", @"Pclass"), new InputOutputColumnPair(@"Age", @"Age"), new InputOutputColumnPair(@"SibSp", @"SibSp"), new InputOutputColumnPair(@"Parch", @"Parch"), new InputOutputColumnPair(@"Fare", @"Fare") }))
                           .Append(context.Transforms.Concatenate(@"TextFeature", @"Name", "Ticket", "Cabin"))
                           .Append(context.Transforms.Text.FeaturizeText("TextFeature", "TextFeature"))
                           .Append(context.Transforms.Concatenate(@"Features", new[] { @"Sex", @"Embarked", @"Pclass", @"Age", @"SibSp", @"Parch", @"Fare", "TextFeature" }))
                           .Append(context.Transforms.Conversion.ConvertType("Survived", "Survived", Data.DataKind.Boolean))
                           .Append(context.Auto().BinaryClassification(labelColumnName: "Survived", useFastForest: false));

            var trainTestSplit = context.Data.TrainTestSplit(df, 0.1);

            var monitor = new MLContextMonitor(context);
            context.Log += (o, e) =>
            {
                if (e.Source.StartsWith("AutoMLExperiment"))
                {
                    Console.WriteLine(e.RawMessage);
                }
            };

            var experiment = context.Auto().CreateExperiment()
                                .SetPipeline(pipeline)
                                .SetTrainingTimeInSeconds(100)
                                .SetDataset(trainTestSplit.TrainSet, trainTestSplit.TestSet)
                                .SetEvaluateMetric(BinaryClassificationMetric.Accuracy, "Survived", "PredictedLabel")
                                .SetTunerFactory(() => new RandomTuner())
                                .SetMonitor(monitor);

            var res = experiment.Run().Result;


            //var tuner = new GridSearchTuner(pipeline.SearchSpace);
            //var df = DataFrame.LoadCsv(TrainDataPath);
            //var trainTestSplit = context.Data.TrainTestSplit(df, 0.1);
            //var bestAccuracy = 0.0;
            //var i = 0;

            //foreach (var param in tuner.Propose())
            //{
            //    Console.WriteLine($"trial {i++}");

            //    var trainingPipeline = pipeline.BuildTrainingPipeline(context, param);
            //    var model = trainingPipeline.Fit(trainTestSplit.TrainSet);
            //    var eval = model.Transform(trainTestSplit.TestSet);
            //    var accuracy = context.BinaryClassification.Evaluate(eval, "Survived").Accuracy;
            //    if (accuracy > bestAccuracy)
            //    {
            //        Console.WriteLine("Found best accuracy");
            //        Console.WriteLine("Current best parameter");
            //        Console.WriteLine(JsonSerializer.Serialize(param));
            //        bestAccuracy = accuracy;
            //    }

            //    Console.WriteLine($"Trial {i}: Current Best Accuracy {bestAccuracy}, Current Accuracy {accuracy}");
            //}

        }

        private class FastTreeOption
        {
            [Range(2, 1024)]
            public int NumberOfTrees { get; set; }

            [Range(2, 1024)]
            public int NumberOfLeaves { get; set; }
        }

        private class FeaturizeTextOption
        {
            [Choice(0, 1, 2)]
            public CaseMode CaseMode { get; set; }

            [BooleanChoice]
            public bool KeepDiacritics { get; set; }

            [BooleanChoice]
            public bool KeepNumbers { get; set; }

            [BooleanChoice]
            public bool KeepPunctuations { get; set; }

            [Option]
            public WordBagEstimatorOption WordBagEstimatorOption { get; set; }
        }

        private class WordBagEstimatorOption
        {
            [Range(1, 10)]
            public int NgramLength { get; set; }

            [BooleanChoice]
            public bool UseAllLengths { get; set; }

            [Choice(0, 1, 2)]
            public WeightingCriteria WeightingCriteria { get; set; }
        }
    }
}
