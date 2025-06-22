// Simple Blog Stats Counter for Listing Pages
class ListingCounters {
    constructor() {
        this.statsUrl = '/assets/blog-stats.json';
        this.stats = null;
        this.init();
    }

    async init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setupCounters());
        } else {
            this.setupCounters();
        }
    }

    async setupCounters() {
        await this.loadStats();
        this.addCountersToListings();
    }

    async loadStats() {
        try {
            const response = await fetch(`${this.statsUrl}?v=${Date.now()}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.stats = await response.json();
        } catch (error) {
            this.stats = { posts: {} };
        }
    }

    addCountersToListings() {
        document.querySelectorAll('.quarto-post').forEach(container => {
            this.addCounterToPost(container);
        });
    }

    addCounterToPost(container) {
        const linkElement = container.querySelector('a[href*="/posts/"]');
        const metadataElement = container.querySelector('.metadata');
        
        if (!linkElement || !metadataElement || container.querySelector('.simple-counters')) return;

        const postSlug = this.extractSlugFromUrl(linkElement.href);
        const postStats = this.getPostStats(postSlug);
        
        this.createCounterElements(metadataElement, postStats);
    }

    extractSlugFromUrl(url) {
        const pathname = new URL(url, window.location.origin).pathname;
        const filename = pathname.split('/').pop();
        return filename.replace('.html', '');
    }

    getPostStats(postSlug) {
        if (this.stats && this.stats.posts && this.stats.posts[postSlug]) {
            return this.stats.posts[postSlug];
        }
        return { views: 0, comments: 0 };
    }

    createCounterElements(metadataElement, stats) {
        const counterContainer = document.createElement('div');
        counterContainer.className = 'simple-counters';
        counterContainer.innerHTML = `${this.formatCount(stats.views)} views<br>${this.formatCount(stats.comments)} comments`;
        metadataElement.appendChild(counterContainer);
    }

    formatCount(count) {
        if (count >= 1000) {
            return (count / 1000).toFixed(1) + 'k';
        }
        return count.toString();
    }
}

// Initialize when page loads
new ListingCounters(); 