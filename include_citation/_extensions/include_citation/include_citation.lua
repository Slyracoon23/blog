-- include_citation.lua
-- Quarto extension shortcode for including citations from .qmd files
-- Usage: {{< include_citation /path/to/citation.qmd >}}

-- Function to read the content of a file
local function read_file(file)
  local f = io.open(file, "r")
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
    -- Get the file path from args
    local file_path = args[1]
    
    if not file_path then
      return pandoc.Str("Error: No file path provided")
    end
    
    -- Remove quotes if they exist
    file_path = file_path:gsub("^%s*[\"'](.+)[\"']%s*$", "%1")
    
    -- Get the base directory of the Quarto project
    local base_dir = quarto.project.directory
    
    -- If file_path is not absolute, make it relative to the base directory
    if not file_path:match("^/") then
      file_path = base_dir .. "/" .. file_path
    end
    
    -- Read the file content
    local content, err = read_file(file_path)
    if not content then
      return pandoc.Str("Error: " .. (err or "Failed to read file"))
    end
    
    -- Parse the content as markdown and return it
    return pandoc.read(content).blocks
  end
}