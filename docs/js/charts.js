/* ================================================
   Content Refresh Opportunity Scoring — Charts & Interactions
   ================================================ */

(function () {
    'use strict';

    // --- Intersection Observer for scroll-reveal animations ---
    const sections = document.querySelectorAll('.section');
    
    // Reset initial state — CSS animation is just the fallback
    sections.forEach(function (section) {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.animation = 'none';
    });

    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.08 });

    sections.forEach(function (section) {
        observer.observe(section);
    });

    // --- Active TOC link on scroll ---
    var tocLinks = document.querySelectorAll('.toc-link');
    var sectionIds = Array.from(tocLinks).map(function (link) {
        return link.getAttribute('href').replace('#', '');
    });

    window.addEventListener('scroll', function () {
        var scrollPos = window.scrollY + 120;

        sectionIds.forEach(function (id, index) {
            var el = document.getElementById(id);
            if (!el) return;
            var top = el.offsetTop;
            var bottom = top + el.offsetHeight;

            if (scrollPos >= top && scrollPos < bottom) {
                tocLinks.forEach(function (link) { link.classList.remove('active'); });
                tocLinks[index].classList.add('active');
            }
        });
    });

    // --- Animate bar widths on scroll into view ---
    var barFills = document.querySelectorAll('.bar-fill');
    barFills.forEach(function (bar) {
        var finalWidth = bar.style.width;
        bar.style.width = '0%';

        var barObserver = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    setTimeout(function () {
                        bar.style.width = finalWidth;
                    }, 100);
                    barObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });

        barObserver.observe(bar);
    });

    // --- Smooth anchor clicks ---
    document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
        anchor.addEventListener('click', function (e) {
            var target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

})();
