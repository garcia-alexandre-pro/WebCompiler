using Newtonsoft.Json.Linq;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace WebCompiler
{
    class BabelCompiler : ICompiler
    {
        //private static Regex _errorRx = new Regex(@".+\.jsx:\s(?<message>.+)\((?<line>[0-9]+):(?<column>[0-9]+)\)", RegexOptions.Compiled);
        private string _path;
        private string _output = string.Empty;
        private string _error = string.Empty;

        public BabelCompiler(string path)
        {
            _path = path;
        }

        public CompilerResult Compile(Config config)
        {
            string inputFile = config.InputFileAbsolute;

            FileInfo info = new FileInfo(inputFile);
            string content = File.ReadAllText(info.FullName);

            CompilerResult result = new CompilerResult
            {
                FileName = info.FullName,
                OriginalContent = content,
            };

            try
            {
                RunCompilerProcess(config, info);

                result.CompiledContent = _output;

                if (_error.Length > 0)
                {
                    JObject json = JObject.Parse(_error);

                    CompilerError ce = new CompilerError
                    {
                        FileName = info.FullName,
                        Message = json["message"].ToString(),
                        ColumnNumber = int.Parse(json["column"].ToString()),
                        LineNumber = int.Parse(json["line"].ToString()),
                        IsWarning = !string.IsNullOrEmpty(_output)
                    };

                    result.Errors.Add(ce);
                }
            }
            catch (Exception ex)
            {
                CompilerError error = new CompilerError
                {
                    FileName = info.FullName,
                    Message = string.IsNullOrEmpty(_error) ? ex.Message : _error,
                    LineNumber = 0,
                    ColumnNumber = 0,
                };

                result.Errors.Add(error);
            }

            return result;
        }

        private void RunCompilerProcess(Config config, FileInfo info)
        {
            string arguments = ConstructArguments(config);

            ProcessStartInfo start = new ProcessStartInfo
            {
                WorkingDirectory = info.Directory.FullName,
                UseShellExecute = false,
                WindowStyle = ProcessWindowStyle.Hidden,
                CreateNoWindow = true,
                FileName = "cmd.exe",
                Arguments = $"/c \"\"{Path.Combine(_path, "node_modules\\.bin\\babel.cmd")}\" {arguments} \"{info.FullName}\"\"",
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };

            start.EnvironmentVariables["PATH"] = _path + ";" + start.EnvironmentVariables["PATH"];

            using (Process p = Process.Start(start))
            {
                var stdout = p.StandardOutput.ReadToEndAsync();
                var stderr = p.StandardError.ReadToEndAsync();
                p.WaitForExit();

                _output = stdout.Result;
                _error = stderr.Result;
            }
        }

        private static string ConstructArguments(Config config)
        {
            //string relative = FileHelpers.MakeRelative(config.GetAbsoluteOutputFile().FullName, config.GetAbsoluteInputFile().FullName);
            string arguments = $"--presets react --out-file \"\"";

            var options = BabelOptions.FromConfig(config);

            if (options.SourceMap || config.SourceMap)
                arguments += " --source-maps inline";

            return arguments;
        }
    }
}
