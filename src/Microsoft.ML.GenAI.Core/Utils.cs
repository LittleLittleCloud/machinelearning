// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.GenAI.Core.Extension;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Microsoft.ML.GenAI.Core;
public static class Utils
{
    public static Tensor ApplyRotaryPosEmb(Tensor input, Tensor cos, Tensor sin)
    {
        using var newScope = NewDisposeScope();
        var shape = input.IntShape().ToArray();
        if (shape[2] != 1)
        {
            var x = input[.., .., .., ..(shape[^1] / 2)];
            var y = input[.., .., .., (shape[^1] / 2)..];

            var xCos = x * cos - y * sin;
            var ySin = x * sin + y * cos;

            return torch.cat([xCos, ySin], -1).MoveToOuterDisposeScope();
        }
        else
        {
            return TorchSharp.GenAIKernel.Rope.BF16(input, cos, sin).MoveToOuterDisposeScope();
        }
    }

    public static Module<Tensor, Tensor> GetActivation(string actFn)
    {
        return actFn switch
        {
            "silu" => nn.SiLU(),
            "relu" => nn.ReLU(),
            "gelu" => nn.GELU(),
            "tanh" => nn.Tanh(),
            "swish" => nn.SiLU(),
            _ => throw new ArgumentException("Invalid activation function", actFn),
        };
    }

    public static Tensor RepeatKV(Tensor x, int nRep)
    {
        var batchSize = x.shape[0];
        var nKVHeads = x.shape[1];
        var seqLen = x.shape[2];
        var headDim = x.shape[3];
        if (nRep == 1)
        {
            return x;
        }

        return x.unsqueeze(2)
                .expand(batchSize, nKVHeads, nRep, seqLen, headDim)
                .reshape(batchSize, nKVHeads * nRep, seqLen, headDim);
    }

    internal static string GetEmbeddedResource(string resourceName)
    {
        // read file content from embedded resource
        var assembly = Assembly.GetCallingAssembly();
        var resourceStream = assembly.GetManifestResourceStream(resourceName);

        if (resourceStream == null)
        {
            throw new ArgumentException("Resource not found", resourceName);
        }

        using var reader = new System.IO.StreamReader(resourceStream);
        return reader.ReadToEnd();
    }
}
