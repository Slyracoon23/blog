-- include_with_indent.lua
-- A Quarto extension that includes content from files with indentation

function include_with_indent(args, kwargs, meta, raw_args, context)
  -- Debug logging
  quarto.log.output("include_with_indent called with path: " .. tostring(args[1]))
  
  -- Log kwargs safely
  local kwargs_str = "{"
  if kwargs then
    for k, v in pairs(kwargs) do
      kwargs_str = kwargs_str .. k .. "=" .. tostring(v) .. ", "
    end
  end
  kwargs_str = kwargs_str .. "}"
  quarto.log.output("kwargs: " .. kwargs_str)
  quarto.log.output("context: " .. tostring(context))
  
  -- Get the file path from the first argument
  local original_path = args[1]
  local file_path = original_path
  
  -- Try multiple approaches to resolve the file path
  local possible_paths = {}
  
  -- 0. Check if it's an absolute path (starts with / or drive letter on Windows)
  local is_absolute = false
  if file_path and (string.sub(file_path, 1, 1) == "/" or 
                   (string.find(file_path, "^%a:[\\/]") ~= nil)) then
    is_absolute = true
    table.insert(possible_paths, file_path)
    quarto.log.output("Added absolute path: " .. file_path)
  end
  
  -- 1. Original path
  if not is_absolute then
    table.insert(possible_paths, file_path)
  end
  
  -- 2. Relative to project root if path starts with /
  if file_path and string.sub(file_path, 1, 1) == "/" and not is_absolute then
    local project_dir = quarto.project.directory and quarto.project.directory()
    if project_dir then
      quarto.log.output("Project directory: " .. project_dir)
      local relative_path = string.sub(file_path, 2)
      local project_relative_path = pandoc.path.join({project_dir, relative_path})
      table.insert(possible_paths, project_relative_path)
      quarto.log.output("Added project-relative path: " .. project_relative_path)
    else
      quarto.log.output("No project directory found")
    end
  end
  
  -- 3. Relative to current document directory
  local doc_dir = nil
  if meta and meta.documentdir then
    doc_dir = pandoc.utils.stringify(meta.documentdir)
    quarto.log.output("Document directory: " .. doc_dir)
    
    -- If path is absolute-style (starts with /), try relative to doc dir
    if string.sub(file_path, 1, 1) == "/" and not is_absolute then
      local relative_path = string.sub(file_path, 2)
      local doc_relative_path = pandoc.path.join({doc_dir, relative_path})
      table.insert(possible_paths, doc_relative_path)
      quarto.log.output("Added doc-relative path: " .. doc_relative_path)
    end
    
    -- Also try direct join with doc dir regardless of whether path starts with /
    if not is_absolute then
      local direct_join_path = pandoc.path.join({doc_dir, file_path})
      table.insert(possible_paths, direct_join_path)
      quarto.log.output("Added direct doc join path: " .. direct_join_path)
    end
  else
    quarto.log.output("No document directory found in meta")
  end
  
  -- Try to read from each possible path
  local success, content
  for _, path in ipairs(possible_paths) do
    quarto.log.output("Trying path: " .. path)
    success, content = pcall(function()
      local file = io.open(path, "r")
      if not file then
        return nil
      end
      local content = file:read("*all")
      quarto.log.output("File read successfully from: " .. path)
      file:close()
      file_path = path  -- Store the successful path
      return content
    end)
    
    if success and content then
      break  -- Found and read the file successfully
    end
  end
  
  -- Get indentation amount from kwargs, default to 2 spaces
  local indent_size = tonumber(kwargs.indent) or 2
  quarto.log.output("Using indent size: " .. tostring(indent_size))
  local indent = string.rep(" ", indent_size)
  
  if not file_path then
    quarto.log.warning("Missing file path for include_with_indent shortcode")
    return pandoc.Null()
  end
  
  if not success or not content then
    quarto.log.warning("Failed to read file from any of the tried paths")
    quarto.log.output("Tried paths: ")
    for _, path in ipairs(possible_paths) do
      quarto.log.output("  - " .. path)
    end
    if not success then
      quarto.log.output("Error: " .. tostring(content))
    end
    return pandoc.Null()
  end
  
  -- Store the include directory for path resolution later
  if quarto.project.directory then
    -- Only attempt if quarto.project is available
    local file_metadata = quarto.project.current_file and quarto.project.current_file()
    if file_metadata then
      file_metadata.include_directory = pandoc.path.directory(file_path)
      quarto.log.output("Set include_directory to: " .. pandoc.path.directory(file_path))
    else
      quarto.log.output("No file metadata available")
    end
  else
    quarto.log.output("quarto.project.directory not available")
  end
  
  -- Add indentation to each line
  local indented_content = ""
  for line in content:gmatch("([^\n]*)\n?") do
    if line ~= "" then
      indented_content = indented_content .. indent .. line .. "\n"
    else
      indented_content = indented_content .. "\n"
    end
  end
  quarto.log.output("Created indented content, length: " .. tostring(#indented_content))
  
  -- Parse the content with Pandoc based on the context
  if context == "block" then
    -- For block context, we process as Markdown and return blocks
    quarto.log.output("Processing as block context")
    local doc = pandoc.read(indented_content)
    quarto.log.output("Number of blocks: " .. tostring(#doc.blocks))
    return doc.blocks
  else
    -- For inline context, try to get the first paragraph's content
    quarto.log.output("Processing as inline context")
    local doc = pandoc.read(indented_content)
    if #doc.blocks > 0 and doc.blocks[1].t == "Para" then
      quarto.log.output("Returning first paragraph content")
      return doc.blocks[1].content
    else
      -- Fallback for non-paragraph content
      quarto.log.output("Fallback to stringified content")
      return { pandoc.Str(indented_content) }
    end
  end
end

-- Return the shortcode handler
return {
  ['include_with_indent'] = include_with_indent
}