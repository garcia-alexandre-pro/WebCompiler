using Newtonsoft.Json.Linq;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace WebCompiler
{
    internal class IcedCoffeeScriptCompiler : ICompiler
    {
        //private static Regex _errorRx = new Regex(":(?<line>[0-9]+):(?<column>[0-9]+).*error: (?<message>.+)", RegexOptions.Compiled);
        private string _path;
        private string _error = string.Empty;
        private string _temp = Path.Combine(Path.GetTempPath(), ".iced-coffee-script");

        public IcedCoffeeScriptCompiler(string path)
        {
            _path = path;
        }

        public CompilerResult Compile(Config config)
        {
            FileInfo info = config.GetAbsoluteInputFile();
            string content = File.ReadAllText(info.FullName);

            CompilerResult result = new CompilerResult
            {
                FileName = info.FullName,
                OriginalContent = content,
            };

            string tempFile = Path.ChangeExtension(Path.Combine(_temp, info.Name), ".js");
            string tempMapFile = tempFile + ".map";

            try
            {
                RunCompilerProcess(config, info);

                if (File.Exists(tempFile))
                {
                    result.CompiledContent = File.ReadAllText(tempFile);

                    var options = IcedCoffeeScriptOptions.FromConfig(config);

                    if (options.SourceMap || config.SourceMap)
                    {
                        if (File.Exists(tempMapFile))
                            result.SourceMap = File.ReadAllText(tempMapFile);
                    }
                }

                if (_error.Length > 0)
                {
                    JObject json = JObject.Parse(_error);

                    CompilerError ce = new CompilerError
                    {
                        FileName = info.FullName,
                        Message = json["message"].ToString(),
                        ColumnNumber = int.Parse(json["column"].ToString()),
                        LineNumber = int.Parse(json["line"].ToString()),
                        //IsWarning = !string.IsNullOrEmpty(_output)
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
            finally
            {
                File.Delete(tempFile);
                File.Delete(tempMapFile);
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
                Arguments = $"/c \"\"{Path.Combine(_path, "node_modules\\.bin\\iced.cmd")}\" {arguments} \"{info.FullName}\"\"",
                StandardErrorEncoding = Encoding.UTF8,
                RedirectStandardError = true,
            };

            start.EnvironmentVariables["PATH"] = _path + ";" + start.EnvironmentVariables["PATH"];

            using (Process p = Process.Start(start))
            {
                var stderr = p.StandardError.ReadToEndAsync();
                p.WaitForExit();

                _error = stderr.Result;
            }
        }

        private string ConstructArguments(Config config)
        {
            string arguments = $" --compile --output \"{_temp}\"";

            var options = IcedCoffeeScriptOptions.FromConfig(config);

            if (options.SourceMap || config.SourceMap)
                arguments += " --map";

            if (options.Bare)
                arguments += " --bare";

            if (!string.IsNullOrEmpty(options.RuntimeMode))
                arguments += " --runtime " + options.RuntimeMode;

            return arguments;
        }
    }
}
