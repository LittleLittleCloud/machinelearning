// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.OpenAI
{
    /// <summary>
    /// Collection of extension methods for <see cref="T:Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers" /> to create instances of TorchSharp trainer components.
    /// </summary>
    /// <remarks>
    /// This requires additional nuget dependencies to link against TorchSharp native dlls. See <see cref="T:Microsoft.ML.Vision.ImageClassificationTrainer"/> for more information.
    /// </remarks>
    public static class OpenAICatalog
    {
        public static GPT3TextClassificationTrainer GPT3TextClassification(
            this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string scoreColumnName = DefaultColumnNames.Score,
            string predictedLabelColumnName = DefaultColumnNames.PredictedLabel,
            string textColumnName = "Sentence1",
            string openAIKey = null)
        {
            return new GPT3TextClassificationTrainer(CatalogUtils.GetEnvironment(catalog)
                , labelColumnName, scoreColumnName, predictedLabelColumnName, textColumnName, openAIKey!);
        }
    }
}
