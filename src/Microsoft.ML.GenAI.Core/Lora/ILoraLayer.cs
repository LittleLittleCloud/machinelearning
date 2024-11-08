// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.GenAI.Core.Lora;

internal interface ILoraLayer
{
    /// <summary>
    /// Get the names of the adapters that are available for this layer.
    /// </summary>
    public IList<string> AdapterNames { get; }

    /// <summary>
    /// return true if the specified adapter is merged into the current layer.
    /// </summary>
    /// <param name="adapterName"></param>
    /// <returns></returns>
    public bool IsMerged(string adapterName);

    /// <summary>
    /// Merge the specified adapter into the current layer.
    /// </summary>
    /// <param name="adapterName"></param>
    public void MergeAdapter(string adapterName);

    /// <summary>
    /// Unmerge the specified adapter from the current layer.
    /// </summary>
    /// <param name="adapterName"></param>
    public void UnmergeAdapter(string adapterName);

    public void AddAdapter(string adapterName, int rank, float loraAlpha, float loraDropout);
}
