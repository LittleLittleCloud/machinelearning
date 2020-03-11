// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator 1.0.0.0
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Azure.MachineLearning.Services.GeneratedOld.Models
{
    using Newtonsoft.Json;
    using System.Linq;

    /// <summary>
    /// Compute nodes information related to a Machine Learning compute. Might
    /// differ for every type of compute.
    /// </summary>
    public partial class ComputeNodesInformation
    {
        /// <summary>
        /// Initializes a new instance of the ComputeNodesInformation class.
        /// </summary>
        public ComputeNodesInformation()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the ComputeNodesInformation class.
        /// </summary>
        /// <param name="nextLink">The continuation token.</param>
        public ComputeNodesInformation(string nextLink = default(string))
        {
            NextLink = nextLink;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets the continuation token.
        /// </summary>
        [JsonProperty(PropertyName = "nextLink")]
        public string NextLink { get; private set; }

    }
}
