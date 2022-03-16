// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.SearchSpace
{
    /// <summary>
    /// interface for all tuners.
    /// </summary>
    public interface ITuner
    {
        /// <summary>
        /// Propose the next <see cref="Parameter"/> from current <paramref name="searchSpace"/>.
        /// </summary>
        Parameter Propose(SearchSpace searchSpace);

        /// <summary>
        /// update tuner using according training result. This can only apply to smart tuners which needs training result to adjust next proposed <see cref="Parameter"/> from <see cref="Propose(SearchSpace)"/>
        /// </summary>
        void Update(Parameter param, double metric, bool isMaximize);
    }
}
