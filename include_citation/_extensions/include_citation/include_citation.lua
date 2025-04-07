-- include_citation.lua
-- Quarto extension shortcode for including citations from .qmd files
-- Usage: {{< include_citation /path/to/citation.qmd >}}

-- Function to read the content of a file
local function read_file(file)
  local f, err = io.open(file, "r")
  if not f then
    return nil, "Could not open file: " .. file
  end
  
  local content = f:read("*all")
  f:close()
  return content
end

-- Return the shortcode handler
return {
  ['include_citation'] = function(args, kwargs, meta) 
    -- Get and validate file path
    local file_path = args[1]
    if not file_path then
      return pandoc.Str("Error: No file path provided")
    end
    
    -- Remove quotes and normalize path
    file_path = file_path:gsub("^%s*[\"'](.+)[\"']%s*$", "%1"):gsub("^/", "")
    
    -- Get the Quarto project directory
    local base_dir = quarto.project and quarto.project.directory
    if not base_dir then
      return pandoc.Str("Error: Could not determine Quarto project directory")
    end
    
    -- Read the file
    local content = read_file(base_dir .. "/" .. file_path)
    if not content then
      return pandoc.Str("Error: Could not open file: " .. file_path)
    end
    
    -- Parse the content as markdown and return it
    return pandoc.read(content).blocks
  end
}