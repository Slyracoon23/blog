document.addEventListener('DOMContentLoaded', function() {
  // Add the main-content class to the quarto container
  const mainContent = document.querySelector('.quarto-container');
  if (mainContent) {
    mainContent.classList.add('main-content');
  }
  
  // Create citation panel if it doesn't exist
  if (!document.getElementById('citation-panel')) {
    const panel = document.createElement('div');
    panel.className = 'citation-panel';
    panel.id = 'citation-panel';
    
    const content = document.createElement('div');
    content.className = 'citation-content';
    content.id = 'citation-content';
    
    // Citation detail view
    const detailView = document.createElement('div');
    detailView.id = 'detail-view';
    
    // Header with title and close button
    const headerContainer = document.createElement('div');
    headerContainer.className = 'citation-details-header';
    
    const detailsHeader = document.createElement('div');
    detailsHeader.id = 'citation-title';
    detailsHeader.textContent = 'Citation details';
    
    const closeButton = document.createElement('a');
    closeButton.className = 'close-panel';
    closeButton.textContent = 'Close';
    closeButton.href = '#';
    
    headerContainer.appendChild(detailsHeader);
    headerContainer.appendChild(closeButton);
    detailView.appendChild(headerContainer);
    
    // Citation content - this is the scrollable part
    const citationText = document.createElement('div');
    citationText.id = 'citation-text';
    citationText.className = 'citation-note';
    detailView.appendChild(citationText);
    
    // Citation navigation
    const citationNav = document.createElement('div');
    citationNav.className = 'citation-navigation';
    
    const navCounter = document.createElement('div');
    navCounter.className = 'nav-counter';
    navCounter.id = 'citation-counter';
    navCounter.textContent = '1 of 7';
    
    const navButtons = document.createElement('div');
    
    const prevButton = document.createElement('button');
    prevButton.className = 'nav-btn';
    prevButton.id = 'prev-citation';
    prevButton.textContent = 'Previous';
    prevButton.disabled = true;
    
    const nextButton = document.createElement('button');
    nextButton.className = 'nav-btn';
    nextButton.id = 'next-citation';
    nextButton.textContent = 'Next';
    
    navButtons.appendChild(prevButton);
    navButtons.appendChild(document.createTextNode(' '));
    navButtons.appendChild(nextButton);
    
    citationNav.appendChild(navCounter);
    citationNav.appendChild(navButtons);
    
    detailView.appendChild(citationNav);
    
    content.appendChild(detailView);
    panel.appendChild(content);
    
    // Append directly to the body for fixed positioning
    document.body.appendChild(panel);
  }
  
  // Get panel and elements
  const panel = document.getElementById('citation-panel');
  const titleEl = document.getElementById('citation-title');
  const content = document.getElementById('citation-text');
  const counter = document.getElementById('citation-counter');
  const prevBtn = document.getElementById('prev-citation');
  const nextBtn = document.getElementById('next-citation');
  const closeBtn = document.querySelector('.close-panel');
  
  // Create hidden citations container
  if (!document.querySelector('.hidden-citations')) {
    const hiddenCitations = document.createElement('div');
    hiddenCitations.className = 'hidden-citations';
    document.body.appendChild(hiddenCitations);
  }
  const hiddenCitations = document.querySelector('.hidden-citations');
  
  // Current citation index and total
  let currentIndex = 0;
  let totalCitations = 0;
  let allCitations = [];
  
  // Handle close button click
  if (closeBtn) {
    closeBtn.addEventListener('click', function(e) {
      e.preventDefault();
      panel.classList.remove('active');
      if (mainContent) {
        mainContent.classList.remove('panel-open');
      }
    });
  }
  
  // Update navigation buttons state
  function updateNavButtons() {
    prevBtn.disabled = currentIndex <= 0;
    nextBtn.disabled = currentIndex >= totalCitations - 1;
    counter.textContent = `${currentIndex + 1} of ${totalCitations}`;
  }
  
  // Adjust panel width on mobile
  function adjustPanelWidth() {
    if (window.innerWidth < 992) {
      panel.style.width = '100%';
    } else {
      panel.style.width = '350px';
    }
  }
  
  // Call on initial load
  adjustPanelWidth();
  
  // Listen for window resize
  window.addEventListener('resize', adjustPanelWidth);
  
  // Process Quarto-generated footnotes
  const footnotesSection = document.querySelector('.footnotes');
  if (footnotesSection) {
    // Find all footnote list items
    const footnoteItems = footnotesSection.querySelectorAll('li[id^="fn"]');
    totalCitations = footnoteItems.length;
    
    // Debug log
    console.log('Found ' + footnoteItems.length + ' footnotes');
    
    footnoteItems.forEach(function(footnoteItem, index) {
      const id = footnoteItem.id;
      const num = id.replace('fn', '');
      allCitations.push(id);
      
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
        citationLink.setAttribute('data-citation-index', index);
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
  } else {
    console.log('No footnotes section found');
  }
  
  // Create citation links for any direct [^1] references
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
      
      // Get the index if this citation already exists in allCitations
      const existingIndex = allCitations.indexOf(targetId);
      if (existingIndex !== -1) {
        citationLink.setAttribute('data-citation-index', existingIndex);
      } else {
        citationLink.setAttribute('data-citation-index', allCitations.length);
        allCitations.push(targetId);
      }
      
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
  
  // Show citation content
  function showCitation(index) {
    if (index >= 0 && index < allCitations.length) {
      currentIndex = index;
      const citationId = allCitations[index];
      const citationContent = document.getElementById(citationId + '-content');
      
      if (citationContent) {
        // Make panel visible
        panel.classList.add('active');
        if (mainContent) {
          mainContent.classList.add('panel-open');
        }
        
        // Set citation content in the panel
        content.innerHTML = citationContent.innerHTML;
        
        // Update navigation
        updateNavButtons();
        
        // Debug
        console.log('Citation panel activated', panel.classList.contains('active'));
      }
    }
  }
  
  // Add click event to all citation links
  document.addEventListener('click', function(e) {
    if (e.target.classList.contains('citation-link')) {
      e.preventDefault();
      
      // Get citation index from data attribute
      const citationIndex = parseInt(e.target.getAttribute('data-citation-index'), 10);
      console.log('Citation link clicked, index:', citationIndex);
      showCitation(citationIndex);
    }
  });
  
  // Navigation button events
  if (prevBtn) {
    prevBtn.addEventListener('click', function() {
      if (currentIndex > 0) {
        showCitation(currentIndex - 1);
      }
    });
  }
  
  if (nextBtn) {
    nextBtn.addEventListener('click', function() {
      if (currentIndex < totalCitations - 1) {
        showCitation(currentIndex + 1);
      }
    });
  }

  // Pre-populate totalCitations if we didn't find footnotes
  if (totalCitations === 0 && allCitations.length > 0) {
    totalCitations = allCitations.length;
  }
}); 