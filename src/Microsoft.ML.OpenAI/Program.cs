// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.OpenAI;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.ML.OpenAI
{
    public class Program
    {
        private class TestSingleSentenceData
        {
            public string Sentence1;
            public string Sentiment;
        }

        public static void Main(string[] args)
        {
            try
            {
                var context = new MLContext();
                context.Log += ContextLog;
                var apiKey = "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";
                var dataView = context.Data.LoadFromEnumerable(
                    new List<TestSingleSentenceData>(new TestSingleSentenceData[] {
                    new TestSingleSentenceData()
                    {
                        Sentence1 = "ultimately feels as flat as the scruffy sands of its titular community .",
                        Sentiment = "Class One"
                    },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "with a sharp script and strong performances",
                         Sentiment = "Class Two"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "that director m. night shyamalan can weave an eerie spell and",
                         Sentiment = "Class Three"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "comfortable",
                         Sentiment = "Class One"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "does have its charms .",
                         Sentiment = "Class Two"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "banal as the telling",
                         Sentiment = "Class Three"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "faithful without being forceful , sad without being shrill , `` a walk to remember '' succeeds through sincerity .",
                         Sentiment = "Class One"
                     },
                     new TestSingleSentenceData()
                     {
                         Sentence1 = "leguizamo 's best movie work so far",
                         Sentiment = "Class Two"
                     }
                    }));

                var pipeline = context.Transforms.Conversion.MapValueToKey("Label", "Sentiment")
                    .Append(context.MulticlassClassification.Trainers.GPT3TextClassification(openAIKey: apiKey))
                    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                var model = pipeline.Fit(dataView);

                var test = model.Transform(dataView);
                var predictedLabels = test.GetColumn<string>("PredictedLabel");

                foreach (var label in predictedLabels)
                {
                    Console.WriteLine(label);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Exception {ex}");
            }
        }

        private static void ContextLog(object sender, LoggingEventArgs e)
        {
            Console.WriteLine(e.Message);
        }
    }
}
