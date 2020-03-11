// <auto-generated>
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for
// license information.
//
// Code generated by Microsoft (R) AutoRest Code Generator 1.0.0.0
// Changes may cause incorrect behavior and will be lost if the code is
// regenerated.
// </auto-generated>

namespace Azure.MachineLearning.Services.Generated.Models
{
    using Newtonsoft.Json;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;

    public partial class SqlDataPath
    {
        /// <summary>
        /// Initializes a new instance of the SqlDataPath class.
        /// </summary>
        public SqlDataPath()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the SqlDataPath class.
        /// </summary>
        public SqlDataPath(string sqlTableName = default(string), string sqlQuery = default(string), string sqlStoredProcedureName = default(string), IList<StoredProcedureParameter> sqlStoredProcedureParams = default(IList<StoredProcedureParameter>))
        {
            SqlTableName = sqlTableName;
            SqlQuery = sqlQuery;
            SqlStoredProcedureName = sqlStoredProcedureName;
            SqlStoredProcedureParams = sqlStoredProcedureParams;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "sqlTableName")]
        public string SqlTableName { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "sqlQuery")]
        public string SqlQuery { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "sqlStoredProcedureName")]
        public string SqlStoredProcedureName { get; set; }

        /// <summary>
        /// </summary>
        [JsonProperty(PropertyName = "sqlStoredProcedureParams")]
        public IList<StoredProcedureParameter> SqlStoredProcedureParams { get; set; }

    }
}
