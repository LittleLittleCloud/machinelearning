﻿using System;
using System.Collections.Generic;
using System.Text;
using ApprovalTests;
using ApprovalTests.Reporters;
using Microsoft.ML.CodeGenerator.Templates.Console;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.CodeGenerator.Tests
{
    public class TemplateTest : BaseTestClass
    {
        public TemplateTest(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        [UseReporter(typeof(DiffReporter))]
        public void TestPredictProgram_WithSampleData()
        {
            var predictProgram = new PredictProgram()
            {
                SampleData = new Dictionary<string, string>()
                {
                    { "key1", "\"key1\"" },
                    { "key2", "\"key2\"" },
                    { "key3", "\"key\\\"3\"" },
                },
                TaskType = "null",
                Features = new List<string>(),
                Namespace = "Namespace",
                LabelName = "LabelName",
                TrainDataPath = "/path",
                Separator = ','
            };
            Approvals.Verify(predictProgram.TransformText());
        }
    }
}
