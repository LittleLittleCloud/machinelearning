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
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Machine Learning compute object wrapped into ARM resource envelope.
    /// </summary>
    public partial class ComputeResource : Resource
    {
        /// <summary>
        /// Initializes a new instance of the ComputeResource class.
        /// </summary>
        public ComputeResource()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the ComputeResource class.
        /// </summary>
        /// <param name="id">Specifies the resource ID.</param>
        /// <param name="name">Specifies the name of the resource.</param>
        /// <param name="identity">The identity of the resource.</param>
        /// <param name="location">Specifies the location of the
        /// resource.</param>
        /// <param name="type">Specifies the type of the resource.</param>
        /// <param name="tags">Contains resource tags defined as key/value
        /// pairs.</param>
        /// <param name="properties">Compute properties</param>
        public ComputeResource(string id = default(string), string name = default(string), Identity identity = default(Identity), string location = default(string), string type = default(string), IDictionary<string, string> tags = default(IDictionary<string, string>), Compute properties = default(Compute))
            : base(id, name, identity, location, type, tags)
        {
            Properties = properties;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets compute properties
        /// </summary>
        [JsonProperty(PropertyName = "properties")]
        public Compute Properties { get; set; }

    }
}
