// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using Xunit;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core.Tests;

public class UtilsTests
{
    [Fact]
    public void ItTestInsertAdapterNameIntoStateDict()
    {
        var stateDicts = new Dictionary<string, Tensor>
        {
            { "baseModel.encoder.layer.0.lora_A.weight", torch.rand(1, 2) },
            { "baseModel.encoder.layer.0.lora_B.weight", torch.rand(1, 2) },
            { "baseModel.encoder.layer.0.lora_A.bias", torch.rand(1) },
            { "baseModel.encoder.layer.0.lora_B.bias", torch.rand(1) },
            { "baseModel.encoder.layer.0.bias", torch.rand(1) },
        };

        var updatedStateDict = Utils.InsertAdapterNameIntoStateDict(stateDicts, "adapter1", "lora_");

        Assert.Equal(5, updatedStateDict.Count);
        Assert.True(updatedStateDict.ContainsKey("baseModel.encoder.layer.0.adapter1.lora_A.weight"));
        Assert.True(updatedStateDict.ContainsKey("baseModel.encoder.layer.0.adapter1.lora_B.weight"));
        Assert.True(updatedStateDict.ContainsKey("baseModel.encoder.layer.0.adapter1.lora_A.bias"));
        Assert.True(updatedStateDict.ContainsKey("baseModel.encoder.layer.0.adapter1.lora_B.bias"));
        Assert.True(updatedStateDict.ContainsKey("baseModel.encoder.layer.0.bias"));
    }
}
