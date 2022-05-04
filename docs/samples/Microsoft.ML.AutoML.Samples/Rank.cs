using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Microsoft.Data.Analysis;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Data;
using Microsoft.ML.SearchSpace;
using Microsoft.ML.Transforms.TimeSeries;
using Microsoft.ML.AutoML.CodeGen;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Trainers.FastTree;

namespace Microsoft.ML.AutoML.Samples
{
    /// <summary>
    /// Time series automl forecasting using Sonar dataset.
    /// </summary>
    public static class Rank
    {
        public static void Run()
        {
            // Load file
            var mlContext = new MLContext();
            var trainDataPath = @"C:\Users\xiaoyuz\Downloads\babbage-similarity-clustered-ranked-grouped-validation.csv";
            var df = DataFrame.LoadCsv(trainDataPath);
            var trainTestSplit = mlContext.Data.TrainTestSplit(df, samplingKeyColumnName: "Group");

            var trainDf = trainTestSplit.TrainSet;
            var testDf = trainTestSplit.TestSet;

            var excludeColumns = new[] { "Group", "Rank" };
            var searchSpace = new SearchSpace<LgbmOption>();
            var featureColumns = df.Columns.Where(c => !excludeColumns.Contains(c.Name) && c.Name.StartsWith("Embedding")).Select(c => c.Name).ToArray();
            var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Group", "Group"))
                .Append(mlContext.Auto().CreateSweepableEstimator((context, param) =>
                {
                    var option = new LightGbmRankingTrainer.Options()
                    {
                        NumberOfLeaves = param.NumberOfLeaves,
                        NumberOfIterations = param.NumberOfTrees,
                        MinimumExampleCountPerLeaf = param.MinimumExampleCountPerLeaf,
                        LearningRate = param.LearningRate,
                        LabelColumnName = "Rank",
                        FeatureColumnName = "Features",
                        MaximumBinCountPerFeature = param.MaximumBinCountPerFeature,
                        RowGroupColumnName = "Group",
                        HandleMissingValue = true,
                    };

                    return mlContext.Ranking.Trainers.LightGbm(option);
                }, searchSpace));

            var autoMLExperiment = mlContext.Auto().CreateExperiment();

            var runner = new RankRunner(mlContext, trainDf, testDf);
            mlContext.Log += (e, o) =>
            {
                if (o.Source.StartsWith("AutoMLExperiment"))
                {
                    Console.WriteLine(o.RawMessage);
                }
            };

            autoMLExperiment.SetPipeline(pipeline)
                            .SetTrialRunner(runner)
                            .SetTrainingTimeInSeconds(6000)
                            .SetIsMaximizeMetric(true);

            var res = autoMLExperiment.Run().Result;
            var bestModel = res.Model;
        }
    }
    public class RankRunner : ITrialRunner
    {
        private MLContext _context;
        private IDataView _train;
        private IDataView _test;

        public RankRunner(MLContext context, IDataView train, IDataView test)
        {
            this._context = context;
            this._train = train;
            this._test = test;
        }

        public TrialResult Run(TrialSettings settings, IServiceProvider provider)
        {
            try
            {
                var trainDataset = this._train;
                var testDataset = this._test;

                var stopWatch = new Stopwatch();
                stopWatch.Start();
                var pipeline = settings.Pipeline.BuildTrainingPipeline(this._context, settings.Parameter);
                var model = pipeline.Fit(trainDataset);
                var eval = model.Transform(testDataset);
                var metric = this._context.Ranking.Evaluate(eval, "Rank", "Group");
                stopWatch.Stop();

                return new TrialResult()
                {
                    Metric = metric.DiscountedCumulativeGains[2],
                    Model = model,
                    TrialSettings = settings,
                    DurationInMilliseconds = stopWatch.ElapsedMilliseconds,
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return new TrialResult()
                {
                    Metric = double.MinValue,
                    Model = null,
                    TrialSettings = settings,
                    DurationInMilliseconds = 0,
                };
            }
        }
    }
}
