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
    /// The root error.
    /// </summary>
    public partial class RootError
    {
        /// <summary>
        /// Initializes a new instance of the RootError class.
        /// </summary>
        public RootError()
        {
            CustomInit();
        }

        /// <summary>
        /// Initializes a new instance of the RootError class.
        /// </summary>
        /// <param name="code">The service-defined error code.</param>
        /// <param name="message">A human-readable representation of the
        /// error.</param>
        /// <param name="target">The target of the error (e.g., the name of the
        /// property in error).</param>
        /// <param name="details">The related errors that occurred during the
        /// request.</param>
        /// <param name="innerError">A nested list of inner errors. When
        /// evaluating errors, clients MUST traverse through all of the nested
        /// “innerErrors” and choose the deepest one that they
        /// understand.</param>
        /// <param name="debugInfo">An internal representation of the error.
        /// May be null for non-AzureML services.</param>
        public RootError(string code = default(string), string message = default(string), string target = default(string), IList<ErrorDetails> details = default(IList<ErrorDetails>), InnerErrorResponse innerError = default(InnerErrorResponse), DebugInfoResponse debugInfo = default(DebugInfoResponse))
        {
            Code = code;
            Message = message;
            Target = target;
            Details = details;
            InnerError = innerError;
            DebugInfo = debugInfo;
            CustomInit();
        }

        /// <summary>
        /// An initialization method that performs custom operations like setting defaults
        /// </summary>
        partial void CustomInit();

        /// <summary>
        /// Gets or sets the service-defined error code.
        /// </summary>
        [JsonProperty(PropertyName = "code")]
        public string Code { get; set; }

        /// <summary>
        /// Gets or sets a human-readable representation of the error.
        /// </summary>
        [JsonProperty(PropertyName = "message")]
        public string Message { get; set; }

        /// <summary>
        /// Gets or sets the target of the error (e.g., the name of the
        /// property in error).
        /// </summary>
        [JsonProperty(PropertyName = "target")]
        public string Target { get; set; }

        /// <summary>
        /// Gets or sets the related errors that occurred during the request.
        /// </summary>
        [JsonProperty(PropertyName = "details")]
        public IList<ErrorDetails> Details { get; set; }

        /// <summary>
        /// Gets or sets a nested list of inner errors. When evaluating errors,
        /// clients MUST traverse through all of the nested “innerErrors” and
        /// choose the deepest one that they understand.
        /// </summary>
        [JsonProperty(PropertyName = "innerError")]
        public InnerErrorResponse InnerError { get; set; }

        /// <summary>
        /// Gets or sets an internal representation of the error. May be null
        /// for non-AzureML services.
        /// </summary>
        [JsonProperty(PropertyName = "debugInfo")]
        public DebugInfoResponse DebugInfo { get; set; }

    }
}
