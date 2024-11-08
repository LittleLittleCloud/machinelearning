// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ApprovalTests;
using ApprovalTests.Namers;
using ApprovalTests.Reporters;
using Microsoft.ML.GenAI.Core.Extension;
using Xunit;

namespace Microsoft.ML.GenAI.Core.Tests;

public class GenAILinearTests
{
    [Fact]
    [UseReporter(typeof(DiffReporter))]
    [UseApprovalSubdirectory("Approvals")]
    public void LoraShapeTests()
    {
        var device = "meta";
        var model = new GenAILinear(100, 200, device: device);

        model.AddAdapter("adapter1", 1, 0.1f, 0.1f);

        var shape = model.PeekShape();

        Approvals.Verify(shape);
    }
}
