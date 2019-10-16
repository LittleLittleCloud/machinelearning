// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Xunit;
using System.Linq;
using Microsoft.ML.RunTests;
using System.IO;
using System;
using Microsoft.ML.AutoML.Tests.Datasets;

namespace Microsoft.ML.AutoML.Test
{
    
    public class AutoFitTests
    {
        [Fact]
        public void AutoFitBinaryTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var result = context.Auto()
                .CreateBinaryClassificationExperiment(0)
                .Execute(trainData, new ColumnInformation() { LabelColumnName = DatasetUtil.UciAdultLabel });
            Assert.True(result.BestRun.ValidationMetrics.Accuracy > 0.70);
            Assert.NotNull(result.BestRun.Estimator);
            Assert.NotNull(result.BestRun.Model);
            Assert.NotNull(result.BestRun.TrainerName);
        }

        [Fact]
        public void AutoFitMultiTest()
        {
            var context = new MLContext();
            var columnInference = context.Auto().InferColumns(DatasetUtil.TrivialMulticlassDatasetPath, DatasetUtil.TrivialMulticlassDatasetLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(DatasetUtil.TrivialMulticlassDatasetPath);
            var result = context.Auto()
                .CreateMulticlassClassificationExperiment(0)
                .Execute(trainData, 5, DatasetUtil.TrivialMulticlassDatasetLabel);
            Assert.True(result.BestRun.Results.First().ValidationMetrics.MicroAccuracy >= 0.7);
            var scoredData = result.BestRun.Results.First().Model.Transform(trainData);
            Assert.Equal(NumberDataViewType.Single, scoredData.Schema[DefaultColumnNames.PredictedLabel].Type);
        }

        [Fact]
        public void AutoFitRegressionTest()
        {
            var context = new MLContext();
            var dataPath = DatasetUtil.DownloadMlNetGeneratedRegressionDataset();
            var columnInference = context.Auto().InferColumns(dataPath, DatasetUtil.MlNetGeneratedRegressionLabel);
            var textLoader = context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
            var trainData = textLoader.Load(dataPath);
            var validationData = context.Data.TakeRows(trainData, 20);
            trainData = context.Data.SkipRows(trainData, 20);
            var result = context.Auto()
                .CreateRegressionExperiment(0)
                .Execute(trainData, validationData,
                    new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel });

            Assert.True(result.RunDetails.Max(i => i.ValidationMetrics.RSquared > 0.9));
        }

        [Fact]
        public void AutoFitRecommendationTest()
        {
            const string matrixColName = "UserId";
            const string matrixRowName = "MovieId";
            MLContext mlContext = new MLContext();

            // STEP 1: Load data
            var trainDataPath = GetDataPath(TestDatasets.trivialRecommendation.trainFilename);
            var testDataPath = GetDataPath(TestDatasets.trivialRecommendation.testFilename);
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<MovieRecommendation>(
                trainDataPath, 
                hasHeader: TestDatasets.trivialRecommendation.fileHasHeader, 
                separatorChar: TestDatasets.trivialRecommendation.fileSeparator);
            IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRecommendation>(
                testDataPath, 
                hasHeader: TestDatasets.trivialRecommendation.fileHasHeader, 
                separatorChar: TestDatasets.trivialRecommendation.fileSeparator);

            // STEP 2: Run AutoML experiment
            ExperimentResult<RegressionMetrics> experimentResult = mlContext.Auto()
                .CreateRecommendationExperiment(5)
                .Execute(trainDataView, testDataView,
                    new ColumnInformation() { 
                        LabelColumnName = "Rating",
                        MatrixColumnIndexColumnName = matrixColName,
                        MatrixRowIndexColumnName = matrixRowName
                    });

            // STEP 3: Print metric from best model
            RunDetail<RegressionMetrics> bestRun = experimentResult.BestRun;
            Assert.True(experimentResult.RunDetails.Count() > 1);
            Assert.NotNull(bestRun.ValidationMetrics);
        }

        private static string GetRepoRoot()
        {
#if NETFRAMEWORK
            string directory = AppDomain.CurrentDomain.BaseDirectory;
#else
            string directory = AppContext.BaseDirectory;
#endif

            while (!Directory.Exists(Path.Combine(directory, ".git")) && directory != null)
            {
                directory = Directory.GetParent(directory).FullName;
            }

            if (directory == null)
            {
                return null;
            }
            return directory;
        }

        public static string GetDataPath(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return null;
            return Path.GetFullPath(Path.Combine(Path.Combine(GetRepoRoot(), "test", "data"), name));
        }
    }
}
