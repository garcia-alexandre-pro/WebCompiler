using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using Microsoft.Build.Construction;
using Newtonsoft.Json;

namespace WebCompiler
{
    /// <summary>
    /// Represents a configuration object used by the compilers.
    /// </summary>
    public class Config
    {
        /// <summary>
        /// The file path to the configuration file.
        /// </summary>
        [JsonIgnore]
        public string FileName { get; set; }

        /// <summary>
        /// The relative file path to the output file.
        /// </summary>
        [JsonProperty("outputFile")]
        public string OutputFile { get; set; }

        /// <summary>
        /// The relative file path to the input file.
        /// </summary>
        [JsonProperty("inputFile")]
        public string InputFileRelative { get; set; }

        /// <summary>
        /// The name of the package (in case the sources come from a nuget package and are referenced using packagereference only).
        /// </summary>
        [JsonProperty("packageName")]
        public string PackageName { get; set; }

        /// <summary>
        /// Settings for the minification.
        /// </summary>
        [JsonProperty("minify")]
        public Dictionary<string, object> Minify { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// If true it makes Visual Studio include the output file in the project.
        /// </summary>
        [DefaultValue(true)]
        [JsonProperty("includeInProject")]
        public bool IncludeInProject { get; set; } = true;

        /// <summary>
        /// If true a source map file is generated for the file types that support it.
        /// </summary>
        [JsonProperty("sourceMap")]
        public bool SourceMap { get; set; }

        /// <summary>
        /// Options specific to each compiler. Based on the inputFile property.
        /// </summary>
        [JsonProperty("options")]
        public Dictionary<string, object> Options { get; set; } = new Dictionary<string, object>();

        internal string Output { get; set; }

        /// <summary>
        /// The retrieved absolute file path to the input file.
        /// </summary>
        public string InputFileAbsolute
        {
            get
            {
                string baseFolder = Path.GetDirectoryName(FileName);
                string inputFile;

                // packagereference packages are stored in a different place, we'll try to retrieve their real location based on what's in the project configuration file
                if (String.IsNullOrEmpty(PackageName))
                {
                    inputFile = Path.Combine(baseFolder, InputFileRelative);

                    if (!File.Exists(inputFile))
                    {
                        throw new FileNotFoundException($"No such file found in {baseFolder}", InputFileRelative);
                    }
                }
                else
                {
                    string projectConfigFile = Directory.GetFiles(baseFolder, "*.csproj", SearchOption.AllDirectories).SingleOrDefault()
                        ?? Directory.GetFiles(baseFolder, "*.vbproj", SearchOption.AllDirectories).SingleOrDefault();

                    if (String.IsNullOrEmpty(projectConfigFile))
                    {
                        throw new FileNotFoundException($"No project configuration file found in {baseFolder}");
                    }

                    ProjectRootElement projectRootElement = ProjectRootElement.Open(projectConfigFile);

                    ProjectItemElement property = projectRootElement.AllChildren.SingleOrDefault(x => x.ElementName == "PackageReference" && ((ProjectItemElement)x).Include == PackageName) as ProjectItemElement;

                    string packageName = property.Include;
                    string packageVersion = (property.AllChildren.SingleOrDefault(x => x.ElementName == "Version") as ProjectMetadataElement)?.Value;

                    if (String.IsNullOrEmpty(packageName) || String.IsNullOrEmpty(packageVersion))
                    {
                        throw new ArgumentException(nameof(PackageName), $"Wrong package name");
                    }

                    string nugetFolder = $"C:/Users/{Environment.UserName}/.nuget/packages/{packageName}/{packageVersion}/contentFiles/";

                    inputFile = Path.Combine(nugetFolder, InputFileRelative);

                    if (!File.Exists(inputFile))
                    {
                        throw new FileNotFoundException($"No such file found in {nugetFolder}", InputFileRelative);
                    }
                }

                return inputFile;
            }
        }

        /// <summary>
        /// Converts the relative input file to an absolute file path.
        /// </summary>
        public FileInfo GetAbsoluteInputFile()
        {
            string folder = new FileInfo(FileName).DirectoryName;

            return new FileInfo(Path.Combine(folder, InputFileRelative.Replace("/", "\\")));
        }

        /// <summary>
        /// Converts the relative output file to an absolute file path.
        /// </summary>
        public FileInfo GetAbsoluteOutputFile()
        {
            string folder = new FileInfo(FileName).DirectoryName;

            return new FileInfo(Path.Combine(folder, OutputFile.Replace("/", "\\")));
        }

        /// <summary>
        /// Checks to see if the input file needs compilation
        /// </summary>
        public bool CompilationRequired()
        {
            FileInfo input = GetAbsoluteInputFile();
            FileInfo output = GetAbsoluteOutputFile();

            if (!output.Exists)
                return true;

            if (input.LastWriteTimeUtc > output.LastWriteTimeUtc)
                return true;

            return HasDependenciesNewerThanOutput(input, output);
        }

        private bool HasDependenciesNewerThanOutput(FileInfo input, FileInfo output)
        {
            string projectRoot = new FileInfo(FileName).DirectoryName;
            Dictionary<string, Dependencies> dependencies = DependencyService.GetDependencies(projectRoot, input.FullName);

            if (dependencies != null)
            {
                string key = input.FullName.ToLowerInvariant();

                return CheckForNewerDependenciesRecursively(key, dependencies, output);
            }

            return false;
        }

        private bool CheckForNewerDependenciesRecursively(string key, Dictionary<string, Dependencies> dependencies, FileInfo output, HashSet<string> checkedDependencies = null)
        {
            if (checkedDependencies == null)
                checkedDependencies = new HashSet<string>();

            checkedDependencies.Add(key);

            if (!dependencies.ContainsKey(key))
                return false;

            foreach (string file in dependencies[key].DependentOn.ToArray())
            {
                if (checkedDependencies.Contains(file))
                    continue;

                FileInfo fileInfo = new FileInfo(file);

                if (!fileInfo.Exists)
                    continue;

                if (fileInfo.LastWriteTimeUtc > output.LastWriteTimeUtc)
                    return true;

                if (CheckForNewerDependenciesRecursively(file, dependencies, output, checkedDependencies))
                    return true;
            }

            return false;
        }

        /// <summary>
        /// Determines if the object is equals to the other object.
        /// </summary>
        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (obj.GetType() != GetType()) return false;
            if (obj == this) return true;

            Config other = (Config)obj;

            return GetHashCode() == other.GetHashCode();
        }

        /// <summary>
        /// Returns the hash code for this Config
        /// </summary>
        public override int GetHashCode()
        {
            return OutputFile.GetHashCode();
        }

        /// <summary>For the JSON.NET serializer</summary>
        public bool ShouldSerializeIncludeInProject()
        {
            Config config = new Config();

            return IncludeInProject != config.IncludeInProject;
        }

        /// <summary>For the JSON.NET serializer</summary>
        public bool ShouldSerializeMinify()
        {
            Config config = new Config();

            return !DictionaryEqual(Minify, config.Minify, null);
        }

        /// <summary>For the JSON.NET serializer</summary>
        public bool ShouldSerializeOptions()
        {
            Config config = new Config();

            return !DictionaryEqual(Options, config.Options, null);
        }

        private static bool DictionaryEqual<TKey, TValue>(
            IDictionary<TKey, TValue> first, IDictionary<TKey, TValue> second,
            IEqualityComparer<TValue> valueComparer)
        {
            if (first == second) return true;
            if ((first == null) || (second == null)) return false;
            if (first.Count != second.Count) return false;

            valueComparer = valueComparer ?? EqualityComparer<TValue>.Default;

            foreach (KeyValuePair<TKey, TValue> kvp in first)
            {
                if (!second.TryGetValue(kvp.Key, out TValue secondValue)) return false;
                if (!valueComparer.Equals(kvp.Value, secondValue)) return false;
            }

            return true;
        }
    }
}
