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

    public partial class StoredProcedureParameter
    {
        /// <summary>
        /// Initializes a new instance of the StoredProcedureParameter class.
        /// </summary>
        public StoredProcedureParameter()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the StoredProcedureParameter class.
        /// </summary>
        /// <param name="type">Possible values include: 'String', 'Int',
        /// 'Decimal', 'Guid', 'Boolean', 'Date'</param>
        public StoredProcedureParameter(string name = default(string), string value = default(string), string type = default(string))
        {
            Name = name;
            Value = value;
            Type = type;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "name")]
        public string Name { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "value")]
        public string Value { get; set; }

        /// <summary>
        /// Gets or sets possible values include: 'String', 'Int', 'Decimal',
        /// 'Guid', 'Boolean', 'Date'
        /// </summary>
        [JsonProperty(PropertyName = "type")]
        public string Type { get; set; }

    }
}
