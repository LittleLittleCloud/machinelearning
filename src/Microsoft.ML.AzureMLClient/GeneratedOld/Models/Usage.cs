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
    /// Describes AML Resource Usage.
    /// </summary>
    public partial class Usage
    {
        /// <summary>
        /// Initializes a new instance of the Usage class.
        /// </summary>
        public Usage()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the Usage class.
        /// </summary>
        /// <param name="unit">An enum describing the unit of usage
        /// measurement. Possible values include: 'Count'</param>
        /// <param name="currentValue">The current usage of the
        /// resource.</param>
        /// <param name="limit">The maximum permitted usage of the
        /// resource.</param>
        /// <param name="name">The name of the type of usage.</param>
        public Usage(UsageUnit? unit = default(UsageUnit?), long? currentValue = default(long?), long? limit = default(long?), UsageName name = default(UsageName))
        {
            Unit = unit;
            CurrentValue = currentValue;
            Limit = limit;
            Name = name;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets an enum describing the unit of usage measurement. Possible
        /// values include: 'Count'
        /// </summary>
        [JsonProperty(PropertyName = "unit")]
        public UsageUnit? Unit { get; private set; }

        /// <summary>
        /// Gets the current usage of the resource.
        /// </summary>
        [JsonProperty(PropertyName = "currentValue")]
        public long? CurrentValue { get; private set; }

        /// <summary>
        /// Gets the maximum permitted usage of the resource.
        /// </summary>
        [JsonProperty(PropertyName = "limit")]
        public long? Limit { get; private set; }

        /// <summary>
        /// Gets the name of the type of usage.
        /// </summary>
        [JsonProperty(PropertyName = "name")]
        public UsageName Name { get; private set; }

    }
}
