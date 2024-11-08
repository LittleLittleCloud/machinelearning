// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.GenAI.Core.Lora;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.Core;
internal class GenAILinear : nn.Module<Tensor, Tensor>, ILoraLayer
{
    private readonly Dictionary<string, bool> _mergedAdapters = new();
    private readonly Dictionary<string, float> _adapterScaling = new();

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    protected Tensor? weight;
    protected Tensor? bias;
    protected readonly int _inFeatures;
    protected readonly int _outFeatures;

    protected ModuleDict<Linear>? lora_A;
    protected ModuleDict<Linear>? lora_B;
    protected ModuleDict<nn.Module<Tensor, Tensor>>? lora_dropout;
#pragma warning restore MSML_GeneralName // This name should be PascalCased

    public IList<string> AdapterNames { get; }

    public GenAILinear(int inFeatures, int outFeatures, bool hasBias = true, ScalarType dtype = ScalarType.Float32, string? device = null)
        : base(nameof(GenAILinear))
    {
        this._inFeatures = inFeatures;
        this._outFeatures = outFeatures;
        device ??= torch.get_default_device().ToString();
        this.weight = torch.zeros(outFeatures, inFeatures, dtype: dtype, device: device);
        this.AdapterNames = new List<string>();
        if (hasBias)
        {
            this.bias = torch.zeros(outFeatures, dtype: dtype, device: device);
        }

        base.RegisterComponents();
    }

#pragma warning disable MSML_GeneralName // This name should be PascalCased
    public override Tensor forward(Tensor input)
#pragma warning restore MSML_GeneralName // This name should be PascalCased
    {
        using var dispose = torch.NewDisposeScope();

        // use float32
        var input2 = input.to_type(ScalarType.Float32);
        var weight2 = this.weight!.to_type(ScalarType.Float32);
        var result = torch.matmul(input2, weight2.t());

        if (this.bias is not null)
        {
            result = result + this.bias.to_type(ScalarType.Float32);
        }

        result.to_type(input.dtype, copy: true);

        if (this.AdapterNames.Count == 0)
        {
            return result.MoveToOuterDisposeScope();
        }

        foreach (var adapterName in this.AdapterNames)
        {
            if (this.IsMerged(adapterName))
            {
                continue;
            }

            var loraA = this.lora_A![adapterName];
            var loraB = this.lora_B![adapterName];
            var loraDropout = this.lora_dropout![adapterName];

            var result2 = loraA.forward(result);
            result2 = loraB.forward(result2);
            result2 = loraDropout.forward(result2);

            result = result + result2 * this._adapterScaling[adapterName];
        }

        return result.MoveToOuterDisposeScope();
    }

    public bool IsMerged(string adapterName)
    {
        return this._mergedAdapters.ContainsKey(adapterName);
    }

    public void MergeAdapter(string adapterName)
    {
        // if this adapter is already merged, return
        if (this.IsMerged(adapterName))
        {
            return;
        }

        // if any of this.weight, lora_A, lora_B, lora_dropout is null, throw exception
        if (this.weight is null || this.lora_A?[adapterName] is null || this.lora_B?[adapterName] is null)
        {
            throw new System.Exception("weight, lora_A, lora_B, lora_dropout must be initialized before merging an adapter");
        }

        // W = W + lora_a * lora_b * scale
        this._mergedAdapters[adapterName] = true;

        this.weight = this.weight + torch.matmul(this.lora_A[adapterName].weight!, this.lora_B[adapterName].weight!) * this._adapterScaling[adapterName];
    }

    public void UnmergeAdapter(string adapterName)
    {
        if (!this.IsMerged(adapterName))
        {
            return;
        }

        // throw exception if any of this.weight, lora_A, lora_B, lora_dropout is null
        if (this.weight is null || this.lora_A?[adapterName] is null || this.lora_B?[adapterName] is null)
        {
            throw new System.Exception("weight, lora_A, lora_B, lora_dropout must be initialized before unmerging an adapter");
        }

        this._mergedAdapters.Remove(adapterName);

        this.weight = this.weight - torch.matmul(this.lora_A[adapterName].weight!, this.lora_B[adapterName].weight!) * this._adapterScaling[adapterName];
    }

    public void AddAdapter(string adapterName, int rank, float loraAlpha, float loraDropout)
    {
        // lora_A, lora_B, lora_dropout must be initialized
        // if not, we initialize them here

        if (this.lora_A is null)
        {
            this.lora_A = new ModuleDict<Linear>();
            this.lora_B = new ModuleDict<Linear>();
            this.lora_dropout = new ModuleDict<nn.Module<Tensor, Tensor>>();
            this.register_module(nameof(this.lora_A), lora_A);
            this.register_module(nameof(this.lora_B), lora_B);
            this.register_module(nameof(this.lora_dropout), lora_dropout);
        }

        if (this.AdapterNames.Contains(adapterName))
        {
            throw new System.Exception($"Adapter {adapterName} already exists");
        }

        this.AdapterNames.Add(adapterName);
        this.lora_A.add_module(adapterName, nn.Linear(this._inFeatures, rank, hasBias: false));
        this.lora_B!.add_module(adapterName, nn.Linear(rank, this._outFeatures, hasBias: false));
        this.lora_dropout!.add_module(adapterName, nn.Dropout(loraDropout));
    }
}
