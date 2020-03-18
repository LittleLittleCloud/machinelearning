﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.AutoML.Test
{
    
    public class ColumnInferenceTests : BaseTestClass
    {
        public ColumnInferenceTests(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void UnGroupReturnsMoreColumnsThanGroup()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext(1);
            var columnInferenceWithoutGrouping = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: false);
            foreach (var col in columnInferenceWithoutGrouping.TextLoaderOptions.Columns)
            {
                Assert.False(col.Source.Length > 1 || col.Source[0].Min != col.Source[0].Max);
            }

            var columnInferenceWithGrouping = context.Auto().InferColumns(dataPath, DatasetUtil.UciAdultLabel, groupColumns: true);
            Assert.True(columnInferenceWithGrouping.TextLoaderOptions.Columns.Count() < columnInferenceWithoutGrouping.TextLoaderOptions.Columns.Count());
        }

        [Fact]
        public void IncorrectLabelColumnThrows()
        {
            var dataPath = DatasetUtil.DownloadUciAdultDataset();
            var context = new MLContext(1);
            Assert.Throws<ArgumentException>(new System.Action(() => context.Auto().InferColumns(dataPath, "Junk", groupColumns: false)));
        }

        [Fact]
        public void LabelIndexOutOfBoundsThrows()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new MLContext(1).Auto().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 100));
        }

        [Fact]
        public void IdentifyLabelColumnThroughIndexWithHeader()
        {
            var result = new MLContext(1).Auto().InferColumns(DatasetUtil.DownloadUciAdultDataset(), 14, hasHeader: true);
            Assert.True(result.TextLoaderOptions.HasHeader);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Source[0].Min == 14 && c.Source[0].Max == 14);
            Assert.Equal("hours-per-week", labelCol.Name);
            Assert.Equal("hours-per-week", result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void IdentifyLabelColumnThroughIndexWithoutHeader()
        {
            var result = new MLContext(1).Auto().InferColumns(DatasetUtil.DownloadIrisDataset(), DatasetUtil.IrisDatasetLabelColIndex);
            Assert.False(result.TextLoaderOptions.HasHeader);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Source[0].Min == DatasetUtil.IrisDatasetLabelColIndex &&
                c.Source[0].Max == DatasetUtil.IrisDatasetLabelColIndex);
            Assert.Equal(DefaultColumnNames.Label, labelCol.Name);
            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void DatasetWithEmptyColumn()
        {
            var result = new MLContext(1).Auto().InferColumns(Path.Combine("TestData", "DatasetWithEmptyColumn.txt"), DefaultColumnNames.Label, groupColumns: false);
            var emptyColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Empty");
            Assert.Equal(DataKind.Single, emptyColumn.DataKind);
        }

        [Fact]
        public void DatasetWithBoolColumn()
        {
            var result = new MLContext(1).Auto().InferColumns(Path.Combine("TestData", "BinaryDatasetWithBoolColumn.txt"), DefaultColumnNames.Label);
            Assert.Equal(3, result.TextLoaderOptions.Columns.Count());

            var boolColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool");
            var boolFeaturesColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "BooleanFeatures");
            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == DefaultColumnNames.Label);

            // ensure non-label Boolean column is detected as R4
            Assert.Equal(DataKind.Single, boolColumn.DataKind);
            Assert.Equal(DataKind.Boolean, boolFeaturesColumn.DataKind);

            Assert.Equal(DataKind.Boolean, labelColumn.DataKind);

            // ensure non-label Boolean column is detected as R4
            Assert.Equal(2, result.ColumnInformation.NumericColumnNames.Count);
            Assert.Equal("Bool", result.ColumnInformation.NumericColumnNames.First());
            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void Should_parse_as_numeric_if_allowBinary_is_false()
        {
            var context = new MLContext(1);
            var filePath = Path.Combine("TestData", "BinaryDatasetWithBoolColumn.txt");
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "Label",
            };
            var result = ColumnInferenceApi.InferColumns(context, filePath, columnInfo, ',', null, null, false, false, true, false);
            Assert.Equal(4, result.TextLoaderOptions.Columns.Count());

            var boolColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool");
            var bool1Column = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool1");
            var bool2Column = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool2");

            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Label");

            Assert.Equal(DataKind.Single, labelColumn.DataKind);
            Assert.Equal(DataKind.Boolean, bool1Column.DataKind);
            Assert.Equal(DataKind.Boolean, bool2Column.DataKind);
            Assert.Equal(DataKind.Single, boolColumn.DataKind);
        }

        [Fact]
        public void Should_parse_as_text_if_allowBinary_is_false()
        {
            var context = new MLContext(1);
            var filePath = Path.Combine("TestData", "BinaryDatasetWithBoolColumn.txt");
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "Bool1",
            };
            var result = ColumnInferenceApi.InferColumns(context, filePath, columnInfo, ',', null, null, false, false, true, false);
            Assert.Equal(4, result.TextLoaderOptions.Columns.Count());

            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Bool1");

            Assert.Equal(DataKind.String, labelColumn.DataKind);
        }

        [Fact]
        public void InferDatasetWithoutHeader()
        {
            var context = new MLContext(1);
            var filePath = Path.Combine("TestData", "DatasetWithoutHeader.txt");
            var columnInfo = new ColumnInformation()
            {
                LabelColumnName = "col0",
                UserIdColumnName = "col1",
                ItemIdColumnName = "col2",
            };
            columnInfo.IgnoredColumnNames.Add("col4");
            var result = ColumnInferenceApi.InferColumns(context, filePath, columnInfo, ',', null, null, false, false, false);
            Assert.Equal(6, result.TextLoaderOptions.Columns.Count());

            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "col0");
            var userColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "col1");
            var itemColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "col2");
            var ignoreColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "col4");

            Assert.Equal(DataKind.String, labelColumn.DataKind);
            Assert.Equal(DataKind.Single, userColumn.DataKind);
            Assert.Equal(DataKind.Single, itemColumn.DataKind);
            Assert.Equal(DataKind.Single, ignoreColumn.DataKind);

            Assert.Single(result.ColumnInformation.CategoricalColumnNames);
            Assert.Empty(result.ColumnInformation.TextColumnNames);
        }

        [Fact]
        public void WhereNameColumnIsOnlyFeature()
        {
            var result = new MLContext(1).Auto().InferColumns(Path.Combine("TestData", "NameColumnIsOnlyFeatureDataset.txt"), DefaultColumnNames.Label);
            Assert.Equal(2, result.TextLoaderOptions.Columns.Count());

            var nameColumn = result.TextLoaderOptions.Columns.First(c => c.Name == "Username");
            var labelColumn = result.TextLoaderOptions.Columns.First(c => c.Name == DefaultColumnNames.Label);
            Assert.Equal(DataKind.String, nameColumn.DataKind);
            Assert.Equal(DataKind.Boolean, labelColumn.DataKind);
            
            Assert.Single(result.ColumnInformation.TextColumnNames);
            Assert.Equal("Username", result.ColumnInformation.TextColumnNames.First());
            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
        }

        [Fact]
        public void DefaultColumnNamesInferredCorrectly()
        {
            var result = new MLContext(1).Auto().InferColumns(Path.Combine("TestData", "DatasetWithDefaultColumnNames.txt"),
                new ColumnInformation()
                {
                    LabelColumnName = DefaultColumnNames.Label,
                    ExampleWeightColumnName = DefaultColumnNames.Weight,
                    UserIdColumnName = DefaultColumnNames.User,
                    ItemIdColumnName = DefaultColumnNames.Item,
                },
                groupColumns : false);

            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
            Assert.Equal(DefaultColumnNames.Weight, result.ColumnInformation.ExampleWeightColumnName);
            Assert.Equal(DefaultColumnNames.User, result.ColumnInformation.UserIdColumnName);
            Assert.Equal(DefaultColumnNames.Item, result.ColumnInformation.ItemIdColumnName);
            Assert.Equal(3, result.ColumnInformation.NumericColumnNames.Count());
        }

        [Fact]
        public void DefaultColumnNamesNoGrouping()
        {
            var result = new MLContext(1).Auto().InferColumns(Path.Combine("TestData", "DatasetWithDefaultColumnNames.txt"),
                new ColumnInformation()
                {
                    LabelColumnName = DefaultColumnNames.Label,
                    ExampleWeightColumnName = DefaultColumnNames.Weight,
                });

            Assert.Equal(DefaultColumnNames.Label, result.ColumnInformation.LabelColumnName);
            Assert.Equal(DefaultColumnNames.Weight, result.ColumnInformation.ExampleWeightColumnName);
            Assert.Single(result.ColumnInformation.NumericColumnNames);
            Assert.Equal(DefaultColumnNames.Features, result.ColumnInformation.NumericColumnNames.First());
        }

        [Fact]
        public void InferColumnsColumnInfoParam()
        {
            var columnInfo = new ColumnInformation() { LabelColumnName = DatasetUtil.MlNetGeneratedRegressionLabel };
            var result = new MLContext(1).Auto().InferColumns(DatasetUtil.DownloadMlNetGeneratedRegressionDataset(), 
                columnInfo);
            var labelCol = result.TextLoaderOptions.Columns.First(c => c.Name == DatasetUtil.MlNetGeneratedRegressionLabel);
            Assert.Equal(DataKind.Single, labelCol.DataKind);
            Assert.Equal(DatasetUtil.MlNetGeneratedRegressionLabel, result.ColumnInformation.LabelColumnName);
            Assert.Single(result.ColumnInformation.NumericColumnNames);
            Assert.Equal(DefaultColumnNames.Features, result.ColumnInformation.NumericColumnNames.First());
            Assert.Null(result.ColumnInformation.ExampleWeightColumnName);
        }
    }
}