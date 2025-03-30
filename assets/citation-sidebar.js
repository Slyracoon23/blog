// Citation Sidebar JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Create citation panel if it doesn't exist
  if (!document.getElementById('citation-panel')) {
    const panel = document.createElement('div');
    panel.className = 'citation-panel';
    panel.id = 'citation-panel';
    
    // Add header to the panel
    const header = document.createElement('div');
    header.className = 'citation-header';
    header.textContent = 'Source References';
    panel.appendChild(header);
    
    const content = document.createElement('div');
    content.className = 'citation-content';
    content.id = 'citation-content';
    
    const citationTitle = document.createElement('h3');
    citationTitle.className = 'citation-title';
    citationTitle.id = 'citation-title';
    citationTitle.textContent = 'Citations';
    
    const citationText = document.createElement('div');
    citationText.id = 'citation-text';
    
    // Add default message
    const defaultMessage = document.createElement('div');
    defaultMessage.className = 'default-citation-message';
    defaultMessage.id = 'default-citation-message';
    defaultMessage.textContent = 'Click on any citation number in the text to view its details here.';
    
    // Add available citations section
    const availableCitations = document.createElement('div');
    availableCitations.className = 'available-citations';
    availableCitations.id = 'available-citations';
    
    const availableTitle = document.createElement('h4');
    availableTitle.textContent = 'Available Citations in this Article:';
    availableCitations.appendChild(availableTitle);
    
    const citationList = document.createElement('ul');
    citationList.className = 'citation-list';
    
    // Add mock citation entries to the list
    const mockCitations = [
      { id: 'fn1', title: 'Anthropic CEO prediction about AI coding (2025)' },
      { id: 'mock-jina', title: 'Jina Reader content enhancement tools' },
      { id: 'mock-rag', title: 'Retrieval-Augmented Generation (RAG) systems' },
      { id: 'mock-fact', title: 'Google Fact Check Explorer capabilities' },
      { id: 'mock-workflow', title: 'Reference tracking workflow methods' }
    ];
    
    mockCitations.forEach(citation => {
      const item = document.createElement('li');
      item.textContent = citation.title;
      item.setAttribute('data-citation-id', citation.id);
      citationList.appendChild(item);
    });
    
    availableCitations.appendChild(citationList);
    
    content.appendChild(citationTitle);
    content.appendChild(defaultMessage);
    content.appendChild(availableCitations);
    content.appendChild(citationText);
    panel.appendChild(content);
    
    document.body.appendChild(panel);
  }
  
  // Get elements
  const panel = document.getElementById('citation-panel');
  const titleEl = document.getElementById('citation-title');
  const content = document.getElementById('citation-text');
  const defaultMessage = document.getElementById('default-citation-message');
  const availableCitations = document.getElementById('available-citations');
  
  // Create hidden citations container
  if (!document.querySelector('.hidden-citations')) {
    const hiddenCitations = document.createElement('div');
    hiddenCitations.className = 'hidden-citations';
    document.body.appendChild(hiddenCitations);
  }
  const hiddenCitations = document.querySelector('.hidden-citations');
  
  // First, handle specifically the Quarto-generated footnotes
  // Quarto usually creates a section with class 'footnotes' at the bottom
  const footnotesSection = document.querySelector('.footnotes');
  if (footnotesSection) {
    // Find all footnote list items
    const footnoteItems = footnotesSection.querySelectorAll('li[id^="fn"]');
    
    footnoteItems.forEach(function(footnoteItem) {
      const id = footnoteItem.id;
      const num = id.replace('fn', '');
      
      // Find the corresponding reference in the text
      const refId = 'fnref' + num;
      const footnoteRef = document.getElementById(refId);
      
      if (footnoteRef) {
        // Create a new citation link
        const citationLink = document.createElement('a');
        citationLink.className = 'citation-link';
        citationLink.textContent = footnoteRef.textContent || num;
        citationLink.setAttribute('data-citation', id);
        citationLink.setAttribute('data-citation-number', num);
        citationLink.href = '#';
        
        // Replace the reference with our citation link
        if (footnoteRef.parentNode) {
          footnoteRef.parentNode.replaceChild(citationLink, footnoteRef);
        }
        
        // Create a citation div for the panel
        const citationDiv = document.createElement('div');
        citationDiv.id = id + '-content';
        citationDiv.className = 'citation-note';
        
        // Copy content but remove the backref
        let contentHtml = footnoteItem.innerHTML;
        
        // Remove the backref link if it exists
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = contentHtml;
        const backRef = tempDiv.querySelector('.footnote-back, a[href^="#fnref"]');
        if (backRef && backRef.parentNode) {
          backRef.parentNode.removeChild(backRef);
        }
        
        citationDiv.innerHTML = tempDiv.innerHTML;
        hiddenCitations.appendChild(citationDiv);
      }
    });
    
    // Hide the footnotes section
    footnotesSection.style.display = 'none';
  }
  
  // Create citation links for any direct [^1] references
  // This handles cases where footnotes might be differently structured
  document.querySelectorAll('a[href^="#fn"], a.footnote-ref').forEach(function(link) {
    // Skip if we've already processed this footnote
    if (link.classList.contains('citation-link') || link.parentNode.classList.contains('replaced-by-citation')) {
      return;
    }
    
    const href = link.getAttribute('href');
    if (!href) return;
    
    const targetId = href.replace('#', '');
    const target = document.getElementById(targetId);
    
    if (target) {
      // Create citation link
      const citationLink = document.createElement('a');
      citationLink.className = 'citation-link';
      citationLink.textContent = link.textContent;
      citationLink.setAttribute('data-citation', targetId);
      citationLink.href = '#';
      
      // Mark the original parent as replaced
      if (link.parentNode) {
        link.parentNode.classList.add('replaced-by-citation');
        link.parentNode.replaceChild(citationLink, link);
      }
      
      // Create hidden citation content if it doesn't exist yet
      if (!document.getElementById(targetId + '-content')) {
        const citationDiv = document.createElement('div');
        citationDiv.id = targetId + '-content';
        citationDiv.className = 'citation-note';
        
        // Copy content but remove the backref
        let targetContent = target.innerHTML;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = targetContent;
        
        const backRef = tempDiv.querySelector('.footnote-back, a[href^="#fnref"]');
        if (backRef && backRef.parentNode) {
          backRef.parentNode.removeChild(backRef);
        }
        
        citationDiv.innerHTML = tempDiv.innerHTML;
        hiddenCitations.appendChild(citationDiv);
      }
    }
  });
  
  // Add click event to all citation links
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('citation-link')) {
      e.preventDefault();
      
      // Get citation ID from data attribute
      const citationId = e.target.getAttribute('data-citation');
      const citationNum = e.target.getAttribute('data-citation-number') || "";
      
      // Get citation content
      const citationContent = document.getElementById(citationId + '-content');
      
      if (citationContent) {
        // Hide default message when showing a citation
        if (defaultMessage) {
          defaultMessage.style.display = 'none';
        }
        
        // Hide available citations list when showing a specific citation
        if (availableCitations) {
          availableCitations.style.display = 'none';
        }
        
        // Set citation title
        titleEl.textContent = "Citation " + citationNum;
        
        // Set citation content in the panel
        content.innerHTML = citationContent.innerHTML;
      }
    }
  });
  
  // Add click event to mock citation list items
  document.querySelectorAll('.citation-list li').forEach(function(item) {
    item.addEventListener('click', function() {
      const citationId = this.getAttribute('data-citation-id');
      
      // For mock citations with no actual content, show a mock content
      if (citationId.startsWith('mock-')) {
        // Hide default message
        if (defaultMessage) {
          defaultMessage.style.display = 'none';
        }
        
        // Hide available citations list
        if (availableCitations) {
          availableCitations.style.display = 'none';
        }
        
        // Set citation title based on the mock citation
        const mockTitle = this.textContent;
        titleEl.textContent = mockTitle;
        
        // Create mock content based on the citation ID
        let mockContent = '';
        
        switch(citationId) {
          case 'mock-jina':
            mockContent = `
              <p>Jina Reader is an AI-powered tool designed for content extraction and analysis.</p>
              <p>Its grounding API provides factuality scores by checking statements against live web results.</p>
              <div class="citation-sources">
                <strong>Sources:</strong>
                <ul>
                  <li><a href="https://jina.ai/reader/" target="_blank">Jina Reader Documentation</a></li>
                  <li><a href="https://example.com/jina-review" target="_blank">Tech Review of Jina Tools</a></li>
                </ul>
              </div>
            `;
            break;
          case 'mock-rag':
            mockContent = `
              <p>Retrieval-Augmented Generation (RAG) combines language modeling with information retrieval, allowing AI systems to access and incorporate external knowledge when generating responses.</p>
              <div class="citation-sources">
                <strong>Sources:</strong>
                <ul>
                  <li><a href="https://example.com/rag-systems" target="_blank">RAG Architecture Overview</a></li>
                  <li><a href="https://example.com/rag-applications" target="_blank">Applications of RAG in Content Creation</a></li>
                </ul>
              </div>
            `;
            break;
          case 'mock-fact':
            mockContent = `
              <p>Google's Fact Check Explorer allows users to search for fact checks on specific topics or claims from various fact-checking organizations around the world.</p>
              <div class="citation-sources">
                <strong>Sources:</strong>
                <ul>
                  <li><a href="https://toolbox.google.com/factcheck/explorer" target="_blank">Google Fact Check Explorer</a></li>
                </ul>
              </div>
            `;
            break;
          case 'mock-workflow':
            mockContent = `
              <p>Reference tracking workflows help content creators maintain the integrity of their research by systematically documenting and verifying sources.</p>
              <p>This includes marking sources as AI-suggested or manually found, and verifying that all references exist and support the information they're cited for.</p>
              <div class="citation-sources">
                <strong>Sources:</strong>
                <ul>
                  <li><a href="https://example.com/reference-tracking" target="_blank">Best Practices in Reference Management</a></li>
                </ul>
              </div>
            `;
            break;
          default:
            mockContent = `<p>Details for this citation are not available.</p>`;
        }
        
        // Set the mock content
        content.innerHTML = mockContent;
      } else if (citationId === 'fn1') {
        // For the actual footnote 1 that exists in the article
        const citationContent = document.getElementById(citationId + '-content');
        if (citationContent) {
          content.innerHTML = citationContent.innerHTML;
        }
      }
    });
  });
}); 