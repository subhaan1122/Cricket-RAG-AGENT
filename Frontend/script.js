/* =============================================
   Cricket World Cup Companion — Enhanced Vanilla JS
   FIXED: Removed duplicate handlers, race conditions, and conflicts
   ============================================= */

// Auto-detect API base URL
var API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? window.location.protocol + '//' + window.location.hostname + ':8000'
  : '';

// Force scroll to top BEFORE the browser restores scroll position
if ('scrollRestoration' in history) {
  history.scrollRestoration = 'manual';
}
window.scrollTo(0, 0);

// Remove any hash from the URL so the browser doesn't auto-scroll to an anchor
if (window.location.hash) {
  history.replaceState(null, '', window.location.pathname + window.location.search);
}

document.addEventListener('DOMContentLoaded', function() {
  // Ensure we're at the top
  window.scrollTo(0, 0);

  // Initialize Lucide icons
  if (typeof lucide !== 'undefined') lucide.createIcons();

  // ── Boot all modules ──
  initNavbar();
  initHamburgerMenu();
  initParticleField('particle-canvas');
  initTypewriter();
  initScrollAnimations();
  initStatsCounters();
  initTournamentTimeline();
  renderPlayerCards();
  renderMatchTimeline();
  initHeroParallax();
  initSmoothScrolling();

  // FIX: Initialize redirect-only chat (no API calls on index page)
  initChatRedirect();

  // Final scroll-to-top after all rendering is done
  requestAnimationFrame(function() {
    window.scrollTo(0, 0);
  });
});

/* =============================================================
   1. NAVBAR — enhanced with better scroll effects
   ============================================================= */
function initNavbar() {
  var nav = document.getElementById('navbar');
  if (!nav) return;

  var lastScrollY = window.scrollY;
  var ticking = false;

  window.addEventListener('scroll', function() {
    if (!ticking) {
      requestAnimationFrame(function() {
        var currentScrollY = window.scrollY;

        // Show/hide based on scroll direction (use classes, not inline styles)
        if (currentScrollY > lastScrollY && currentScrollY > 100) {
          nav.classList.add('nav-hidden');
          nav.classList.remove('nav-visible');
        } else {
          nav.classList.remove('nav-hidden');
          nav.classList.add('nav-visible');
        }

        // Glass effect on scroll
        nav.classList.toggle('scrolled', currentScrollY > 50);

        // Highlight active section
        highlightActiveNavLink(currentScrollY);

        lastScrollY = currentScrollY;
        ticking = false;
      });
      ticking = true;
    }
  });
}

/* NEW: Mobile hamburger menu */
function initHamburgerMenu() {
  var toggle = document.getElementById('navbar-toggle');
  var links = document.getElementById('navbar-links');
  if (!toggle || !links) return;

  toggle.addEventListener('click', function() {
    toggle.classList.toggle('active');
    links.classList.toggle('open');
  });

  // Close menu when a link is clicked
  links.querySelectorAll('a').forEach(function(link) {
    link.addEventListener('click', function() {
      toggle.classList.remove('active');
      links.classList.remove('open');
    });
  });
}

function highlightActiveNavLink(scrollY) {
  var sections = ['hero', 'timeline', 'players', 'matches', 'chat'];
  var navLinks = document.querySelectorAll('.navbar-links a');

  var currentSection = 'hero';

  sections.forEach(function(section) {
    var element = document.getElementById(section);
    if (element) {
      var rect = element.getBoundingClientRect();
      if (rect.top <= 100 && rect.bottom >= 100) {
        currentSection = section;
      }
    }
  });

  navLinks.forEach(function(link) {
    link.classList.remove('active');
    if (link.getAttribute('href') === '#' + currentSection) {
      link.classList.add('active');
    }
  });
}

/* =============================================================
   2. PARTICLE FIELD — with throttled rendering & mouse fix
   ============================================================= */
function initParticleField(canvasId) {
  var canvas = document.getElementById(canvasId);
  if (!canvas) return;

  var ctx = canvas.getContext('2d');
  var particles = [];
  var animationId;
  var mouse = { x: -9999, y: -9999, radius: 100 };

  function resizeCanvas() {
    var dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    initParticlesData();
  }

  function initParticlesData() {
    particles = [];
    var w = canvas.offsetWidth;
    var h = canvas.offsetHeight;
    var particleCount = 80;

    for (var i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        size: Math.random() * 3 + 1,
        speedX: (Math.random() - 0.5) * 0.3,
        speedY: (Math.random() - 0.5) * 0.3,
        color: 'hsla(' + (42 + Math.random() * 30) + ', 55%, 55%, ' + (Math.random() * 0.4 + 0.1) + ')'
      });
    }
  }

  function drawParticles() {
    var w = canvas.offsetWidth;
    var h = canvas.offsetHeight;
    ctx.clearRect(0, 0, w, h);

    particles.forEach(function(p) {
      p.x += p.speedX;
      p.y += p.speedY;

      if (p.x <= 0 || p.x >= w) p.speedX *= -1;
      if (p.y <= 0 || p.y >= h) p.speedY *= -1;

      // Clamp to bounds
      p.x = Math.max(0, Math.min(w, p.x));
      p.y = Math.max(0, Math.min(h, p.y));

      // Mouse interaction
      var dx = mouse.x - p.x;
      var dy = mouse.y - p.y;
      var distance = Math.sqrt(dx * dx + dy * dy);

      if (distance < mouse.radius && distance > 0) {
        var angle = Math.atan2(dy, dx);
        var force = (mouse.radius - distance) / mouse.radius;
        p.x -= Math.cos(angle) * force * 2;
        p.y -= Math.sin(angle) * force * 2;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    });

    // Connect nearby particles
    for (var i = 0; i < particles.length; i++) {
      for (var j = i + 1; j < particles.length; j++) {
        var ddx = particles[i].x - particles[j].x;
        var ddy = particles[i].y - particles[j].y;
        var dist = Math.sqrt(ddx * ddx + ddy * ddy);

        if (dist < 100) {
          ctx.beginPath();
          ctx.strokeStyle = 'hsla(42, 55%, 55%, ' + (0.1 * (1 - dist / 100)) + ')';
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    animationId = requestAnimationFrame(drawParticles);
  }

  // FIX: Mouse events on the canvas parent (hero section) so pointer-events work
  var parentSection = canvas.closest('section') || canvas.parentElement;
  if (parentSection) {
    parentSection.addEventListener('mousemove', function(e) {
      var rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    });
    parentSection.addEventListener('mouseleave', function() {
      mouse.x = -9999;
      mouse.y = -9999;
    });
  }

  // Debounced resize
  var resizeTimeout;
  window.addEventListener('resize', function() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(resizeCanvas, 200);
  });

  resizeCanvas();
  drawParticles();
}

/* =============================================================
   3. TYPEWRITER — enhanced with better typing effect
   ============================================================= */
function initTypewriter() {
  var textElement = document.getElementById('typewriter-text');
  var cursorElement = document.getElementById('typewriter-cursor');
  if (!textElement || !cursorElement) return;

  var texts = [
    'Who scored the most runs in 2023?',
    "Tell me about Dhoni's winning six",
    'Best bowling figures in World Cups',
    'Most dramatic final ever played',
    "Sachin's World Cup journey",
    'Australia dominance across decades'
  ];

  var textIndex = 0;
  var charIndex = 0;
  var isDeleting = false;
  var isPaused = false;
  var typingSpeed = 60;
  var deletingSpeed = 30;
  var pauseTime = 2000;

  function type() {
    var currentText = texts[textIndex];

    if (!isDeleting && charIndex === currentText.length) {
      if (!isPaused) {
        isPaused = true;
        setTimeout(function() {
          isPaused = false;
          isDeleting = true;
          type();
        }, pauseTime);
      }
      return;
    }

    if (isDeleting && charIndex === 0) {
      isDeleting = false;
      textIndex = (textIndex + 1) % texts.length;
    }

    charIndex += isDeleting ? -1 : 1;

    textElement.textContent = currentText.substring(0, charIndex);

    // Smooth cursor animation
    cursorElement.style.opacity = isPaused ? '0.5' : '1';

    setTimeout(type, isDeleting ? deletingSpeed : typingSpeed);
  }

  setTimeout(type, 1000);
}

/* =============================================================
   4. SCROLL ANIMATIONS — enhanced IntersectionObserver
   ============================================================= */
function initScrollAnimations() {
  if (!('IntersectionObserver' in window)) return;

  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        var delay = parseInt(entry.target.dataset.delay || '0', 10);
        setTimeout(function() {
          entry.target.classList.add('visible');
        }, delay);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.anim-on-scroll').forEach(function(el) {
    observer.observe(el);
  });
}

/* =============================================================
   5. STATS COUNTER — enhanced counting animation
   ============================================================= */
function initStatsCounters() {
  if (!('IntersectionObserver' in window)) return;

  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        var valueElement = entry.target;
        var target = parseInt(valueElement.dataset.target, 10);
        var suffix = valueElement.dataset.suffix || '';
        var card = valueElement.closest('.stats-card');
        var delay = card ? parseInt(card.dataset.delay || '0', 10) : 0;

        setTimeout(function() {
          countUp(valueElement, target, suffix);
        }, delay);

        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.stats-value').forEach(function(el) {
    observer.observe(el);
  });
}

function countUp(element, target, suffix, duration) {
  duration = duration || 2000;
  var startTime = Date.now();

  function update() {
    var elapsed = Date.now() - startTime;
    var progress = Math.min(elapsed / duration, 1);
    var eased = 1 - Math.pow(1 - progress, 3);
    var current = Math.floor(eased * target);

    element.textContent = current.toLocaleString() + suffix;

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      // Subtle completion highlight
      element.style.color = 'var(--secondary)';
      setTimeout(function() {
        element.style.color = '';
      }, 500);
    }
  }

  update();
}

/* =============================================================
   6. HERO PARALLAX — enhanced with smooth effects
   ============================================================= */
function initHeroParallax() {
  var hero = document.getElementById('hero');
  if (!hero) return;

  var bg = hero.querySelector('.hero-bg');
  var content = hero.querySelector('.hero-inner');
  var indicator = hero.querySelector('.scroll-indicator');
  var ticking = false;

  window.addEventListener('scroll', function() {
    if (!ticking) {
      requestAnimationFrame(function() {
        var scrolled = window.pageYOffset;

        if (bg) {
          bg.style.transform = 'translateY(' + (scrolled * -0.25) + 'px)';
        }

        if (content) {
          var opacity = 1 - (scrolled / 500);
          content.style.opacity = Math.max(opacity, 0);
          content.style.transform = 'translateY(' + (scrolled * -0.15) + 'px)';
        }

        if (indicator) {
          indicator.style.opacity = scrolled > 100 ? '0' : '1';
        }

        ticking = false;
      });
      ticking = true;
    }
  });
}

/* =============================================================
   7. SMOOTH SCROLLING — enhanced navigation
   ============================================================= */
function initSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();

      var targetId = this.getAttribute('href');
      if (targetId === '#') return;

      var targetElement = document.querySelector(targetId);
      if (targetElement) {
        var nav = document.querySelector('.navbar');
        var headerHeight = nav ? nav.offsetHeight : 0;
        var targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - headerHeight;

        window.scrollTo({
          top: targetPosition,
          behavior: 'smooth'
        });

        // Update active nav link
        document.querySelectorAll('.navbar-links a').forEach(function(link) {
          link.classList.remove('active');
        });
        if (this.closest('.navbar-links')) {
          this.classList.add('active');
        }
      }
    });
  });
}

/* =============================================================
   8. TOURNAMENT TIMELINE — enhanced with better animations
   ============================================================= */
function initTournamentTimeline() {
  var tournaments = [
    { year: 2003, winner: 'Australia', host: 'South Africa', emoji: '🇦🇺' },
    { year: 2007, winner: 'Australia', host: 'West Indies', emoji: '🇦🇺' },
    { year: 2011, winner: 'India', host: 'India/SL/BD', emoji: '🇮🇳' },
    { year: 2015, winner: 'Australia', host: 'AUS/NZ', emoji: '🇦🇺' },
    { year: 2019, winner: 'England', host: 'England', emoji: '🏴' },
    { year: 2023, winner: 'Australia', host: 'India', emoji: '🇦🇺' },
  ];

  var details = {
    2003: {
      winner: 'Australia',
      runnerUp: 'India',
      venue: 'Johannesburg, SA',
      highlight: "Ponting's 140* in the final",
      motm: 'Ricky Ponting',
      score: '359/2 vs 234',
      fact: 'Sachin scored 673 runs - tournament best'
    },
    2007: {
      winner: 'Australia',
      runnerUp: 'Sri Lanka',
      venue: 'Bridgetown, WI',
      highlight: 'Hat-trick of titles',
      motm: 'Adam Gilchrist',
      score: '281/4 vs 215/8',
      fact: 'First World Cup with Super 8 stage'
    },
    2011: {
      winner: 'India',
      runnerUp: 'Sri Lanka',
      venue: 'Mumbai, India',
      highlight: "Dhoni's iconic winning six",
      motm: 'MS Dhoni',
      score: '275/6 vs 274/6',
      fact: "India's first WC win in 28 years"
    },
    2015: {
      winner: 'Australia',
      runnerUp: 'New Zealand',
      venue: 'Melbourne, AUS',
      highlight: "Starc's 22 wickets",
      motm: 'James Faulkner',
      score: '186/3 vs 183',
      fact: 'Martin Guptill scored 237* - WC record'
    },
    2019: {
      winner: 'England',
      runnerUp: 'New Zealand',
      venue: "Lord's, England",
      highlight: 'Super Over drama',
      motm: 'Ben Stokes',
      score: '241 vs 241 → Super Over',
      fact: 'Won on boundary count rule'
    },
    2023: {
      winner: 'Australia',
      runnerUp: 'India',
      venue: 'Ahmedabad, India',
      highlight: "Head's match-winning century",
      motm: 'Travis Head',
      score: '241/4 vs 240',
      fact: "Kohli broke Sachin's ODI century record"
    },
  };

  var selector = document.getElementById('timeline-selector');
  var detailsPanel = document.getElementById('tournament-details');
  if (!selector || !detailsPanel) return;

  var selectedYear = null;
  var isAutoSelect = true;

  selector.innerHTML = '';

  tournaments.forEach(function(t, i) {
    var btn = document.createElement('button');
    btn.className = 'timeline-btn';
    btn.innerHTML =
      '<span class="trophy-icon"><i data-lucide="trophy" style="width:20px;height:20px;"></i></span>' +
      '<span class="emoji">' + t.emoji + '</span>' +
      '<span class="year">' + t.year + '</span>' +
      '<span class="winner-name">' + t.winner + '</span>';

    setTimeout(function() {
      btn.classList.add('animate-in');
    }, 100 + i * 100);

    btn.addEventListener('click', function() {
      // Deselect others
      selector.querySelectorAll('.timeline-btn').forEach(function(b) {
        b.classList.remove('active');
      });

      btn.classList.add('active');
      showTournamentDetails(t.year, details[t.year]);

      // Scroll into view on mobile (only on user clicks, not auto-select)
      if (window.innerWidth < 768 && !isAutoSelect) {
        setTimeout(function() {
          detailsPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 300);
      }
    });

    selector.appendChild(btn);
  });

  // Auto-select first tournament
  setTimeout(function() {
    if (selector.firstChild) {
      selector.firstChild.click();
    }
    setTimeout(function() { isAutoSelect = false; }, 200);
  }, 1000);

  if (typeof lucide !== 'undefined') lucide.createIcons();

  function showTournamentDetails(year, data) {
    if (selectedYear === year) return;
    selectedYear = year;

    if (!detailsPanel.classList.contains('hidden')) {
      detailsPanel.classList.add('closing');
      setTimeout(function() {
        renderDetails(year, data);
        detailsPanel.classList.remove('closing');
      }, 300);
    } else {
      detailsPanel.classList.remove('hidden');
      renderDetails(year, data);
    }
  }

  function renderDetails(year, data) {
    var hostDisplay = data.venue.split(',')[1] ? data.venue.split(',')[1].trim() : data.venue;
    detailsPanel.innerHTML =
      '<div class="tournament-details-header">' +
        '<i data-lucide="trophy" style="width:32px;height:32px;color:var(--secondary)"></i>' +
        '<h3>' + year + ' ICC Cricket World Cup</h3>' +
      '</div>' +
      '<div class="tournament-details-grid">' +
        '<div><span class="detail-label">Champion</span><span class="detail-value">' + data.winner + ' 🏆</span></div>' +
        '<div><span class="detail-label">Runner-up</span><span class="detail-value">' + data.runnerUp + '</span></div>' +
        '<div><span class="detail-label">Host Nation</span><span class="detail-value">' + hostDisplay + '</span></div>' +
        '<div><span class="detail-label">Final Score</span><span class="detail-value">' + data.score + '</span></div>' +
        '<div><span class="detail-label">Player of the Match</span><span class="detail-value">' + data.motm + '</span></div>' +
        '<div><span class="detail-label">Key Highlight</span><span class="detail-highlight">' + data.highlight + '</span></div>' +
        '<div style="grid-column: 1 / -1; padding: 1.5rem; background: hsla(42,55%,55%,0.05);">' +
          '<span class="detail-label">Did You Know?</span>' +
          '<span class="detail-value" style="font-size: 0.95rem; line-height: 1.6;">' + data.fact + '</span>' +
        '</div>' +
      '</div>';

    // Re-trigger animation
    detailsPanel.style.animation = 'none';
    detailsPanel.offsetHeight; // force reflow
    detailsPanel.style.animation = '';

    if (typeof lucide !== 'undefined') lucide.createIcons();
  }
}

/* =============================================================
   9. PLAYER CARDS — enhanced with hover effects
   ============================================================= */
function renderPlayerCards() {
  var players = [
    {
      name: 'Sachin Tendulkar',
      country: 'India',
      role: 'Batsman',
      runs: 2278,
      wickets: null,
      matches: 45,
      highlight: 'Most World Cup runs ever',
      emoji: '🇮🇳',
      era: '1989-2012'
    },
    {
      name: 'Ricky Ponting',
      country: 'Australia',
      role: 'Batsman',
      runs: 1743,
      wickets: null,
      matches: 46,
      highlight: 'Captained 2 World Cup wins',
      emoji: '🇦🇺',
      era: '1995-2012'
    },
    {
      name: 'Wasim Akram',
      country: 'Pakistan',
      role: 'Bowler',
      runs: null,
      wickets: 55,
      matches: 38,
      highlight: 'Most wickets for Pakistan in WCs',
      emoji: '🇵🇰',
      era: '1984-2003'
    },
    {
      name: 'Ben Stokes',
      country: 'England',
      role: 'All-rounder',
      runs: 1104,
      wickets: 24,
      matches: 27,
      highlight: 'Player of the Match in 2019 final',
      emoji: '🏴',
      era: '2011-Present'
    },
    {
      name: 'Kane Williamson',
      country: 'New Zealand',
      role: 'Batsman',
      runs: 1032,
      wickets: null,
      matches: 25,
      highlight: 'Led NZ to consecutive finals',
      emoji: '🇳🇿',
      era: '2010-Present'
    },
    {
      name: 'AB de Villiers',
      country: 'South Africa',
      role: 'Batsman',
      runs: 1207,
      wickets: null,
      matches: 23,
      highlight: 'Fastest 150 in World Cup history',
      emoji: '🇿🇦',
      era: '2004-2018'
    },
  ];

  var grid = document.getElementById('players-grid');
  if (!grid) return;

  grid.innerHTML = '';

  players.forEach(function(p, i) {
    var card = document.createElement('div');
    card.className = 'player-card anim-on-scroll';
    card.dataset.delay = String(i * 100);

    var statsHtml = '';
    if (p.runs !== null) {
      statsHtml += '<div class="player-stat-box"><span class="stat-label">Runs</span><span class="stat-number">' + p.runs.toLocaleString() + '</span></div>';
    }
    if (p.wickets !== null) {
      statsHtml += '<div class="player-stat-box"><span class="stat-label">Wickets</span><span class="stat-number">' + p.wickets + '</span></div>';
    }
    statsHtml += '<div class="player-stat-box"><span class="stat-label">Matches</span><span class="stat-number">' + p.matches + '</span></div>';
    statsHtml += '<div class="player-stat-box"><span class="stat-label">World Cups</span><span class="stat-number">' + Math.ceil(p.matches / 7) + '</span></div>';

    card.innerHTML =
      '<div class="orb"></div>' +
      '<div class="card-body">' +
        '<div class="player-header">' +
          '<span class="player-emoji">' + p.emoji + '</span>' +
          '<div>' +
            '<div class="player-name">' + p.name + '</div>' +
            '<div class="player-meta">' + p.country + ' • ' + p.role + ' • ' + p.era + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="player-stats">' + statsHtml + '</div>' +
        '<div class="player-highlight"><span>★</span><span>' + p.highlight + '</span></div>' +
      '</div>';

    grid.appendChild(card);
  });

  // Re-observe newly added scroll elements
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var delay = parseInt(entry.target.dataset.delay || '0', 10);
          setTimeout(function() {
            entry.target.classList.add('visible');
          }, delay);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });

    grid.querySelectorAll('.anim-on-scroll').forEach(function(el) {
      observer.observe(el);
    });
  }
}

/* =============================================================
   10. MATCH TIMELINE — enhanced with better animations
   ============================================================= */
function renderMatchTimeline() {
  var matches = [
    {
      year: 2003,
      match: 'Final: Australia vs India',
      result: 'AUS won by 125 runs',
      score: '359/2 vs 234',
      moment: "Ponting's 140* masterclass",
      icon: 'trophy'
    },
    {
      year: 2007,
      match: 'Super 8: Ireland vs Pakistan',
      result: 'IRE won by 3 wickets',
      score: '132 vs 132/7',
      moment: 'Biggest upset in WC history',
      icon: 'zap'
    },
    {
      year: 2011,
      match: 'Final: India vs Sri Lanka',
      result: 'IND won by 6 wickets',
      score: '275/6 vs 274/6',
      moment: 'Dhoni finishes it with a six!',
      icon: 'target'
    },
    {
      year: 2015,
      match: 'QF: NZ vs SA',
      result: 'NZ won by 4 wickets',
      score: '298/6 vs 299/6',
      moment: "Grant Elliott's last-ball six",
      icon: 'heart'
    },
    {
      year: 2019,
      match: 'Final: England vs New Zealand',
      result: 'ENG won (boundary count)',
      score: '241 vs 241 → Super Over tied!',
      moment: 'Greatest final ever played',
      icon: 'award'
    },
    {
      year: 2023,
      match: 'Final: India vs Australia',
      result: 'AUS won by 6 wickets',
      score: '240 vs 241/4',
      moment: "Travis Head's stunning century",
      icon: 'star'
    },
  ];

  var container = document.getElementById('match-timeline');
  if (!container) return;

  // Remove old match items but keep the line
  var existingItems = container.querySelectorAll('.match-item');
  existingItems.forEach(function(el) { el.remove(); });

  matches.forEach(function(m, i) {
    var item = document.createElement('div');
    var direction = i % 2 === 0 ? 'even' : 'odd';
    item.className = 'match-item ' + direction + ' anim-on-scroll';
    item.dataset.delay = String(i * 150);

    item.innerHTML =
      '<div class="match-card" style="flex:1">' +
        '<div class="match-card-header">' +
          '<span class="match-year">' + m.year + '</span>' +
          '<span class="match-dot-sep">•</span>' +
          '<span class="match-name">' + m.match + '</span>' +
        '</div>' +
        '<p class="match-result">' + m.result + '</p>' +
        '<p class="match-score">' + m.score + '</p>' +
        '<div class="match-moment">' +
          '<i data-lucide="' + m.icon + '" style="width:16px;height:16px;color:var(--secondary)"></i>' +
          '<span>' + m.moment + '</span>' +
        '</div>' +
      '</div>' +
      '<div class="match-center-dot"></div>' +
      '<div class="match-spacer"></div>';

    container.appendChild(item);
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();

  // Observer for staggered entrance
  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var delay = parseInt(entry.target.dataset.delay || '0', 10);
          setTimeout(function() {
            entry.target.classList.add('visible');
          }, delay);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.2 });

    container.querySelectorAll('.anim-on-scroll').forEach(function(el) {
      observer.observe(el);
    });
  }
}

/* =============================================================
   11. CHAT REDIRECT — Index page chat redirects to chatbot.html
   FIX: No longer creates API-calling chat; purely redirect-based
   ============================================================= */
function initChatRedirect() {
  var messagesContainer = document.getElementById('chat-messages');
  var form = document.getElementById('chat-form');
  var input = document.getElementById('chat-input');
  var sendBtn = document.getElementById('chat-send-btn');
  var suggestionsContainer = document.getElementById('chat-suggestions');

  if (!messagesContainer || !form) return;

  var suggestedQuestions = [
    'Who won the 2011 World Cup?',
    'Tell me about the 2019 final',
    "Virat Kohli's World Cup records",
    "MS Dhoni's best moments",
    "Australia's 2023 victory",
    "Sachin Tendulkar's legacy",
    'Best bowling figures in World Cup',
    'Most runs in a single tournament'
  ];

  // Show welcome message
  messagesContainer.innerHTML = '';
  var welcomeMsg = document.createElement('div');
  welcomeMsg.className = 'chat-msg bot';
  welcomeMsg.innerHTML =
    '<div class="chat-msg-avatar"><i data-lucide="bot" style="width:14px;height:14px;color:var(--accent)"></i></div>' +
    '<div class="chat-msg-bubble"><div class="chat-msg-text">Welcome to Cricket World Cup Companion! 🏏 I\'m your expert on ICC World Cups from 2003 to 2023. Ask me anything about the tournaments, players, matches, or statistics!</div></div>';
  messagesContainer.appendChild(welcomeMsg);

  // Render suggestions (redirect on click)
  suggestionsContainer.innerHTML = '';
  suggestedQuestions.forEach(function(q) {
    var btn = document.createElement('button');
    btn.className = 'suggestion-btn';
    btn.textContent = q;
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      if (typeof window.redirectToChatbot === 'function') {
        window.redirectToChatbot(q);
      } else {
        window.location.href = 'chatbot.html?q=' + encodeURIComponent(q);
      }
    });
    suggestionsContainer.appendChild(btn);
  });

  // Form submission redirects to chatbot
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    var text = input.value.trim();
    if (text) {
      if (typeof window.redirectToChatbot === 'function') {
        window.redirectToChatbot(text);
      } else {
        window.location.href = 'chatbot.html?q=' + encodeURIComponent(text);
      }
    }
  });

  // Enable/disable send button
  input.addEventListener('input', function() {
    var hasText = input.value.trim().length > 0;
    sendBtn.disabled = !hasText;
    sendBtn.style.opacity = hasText ? '1' : '0.5';
  });

  if (typeof lucide !== 'undefined') lucide.createIcons();
}

/* =============================================================
   12. SHARED MARKDOWN FORMATTER — used by both pages
   ============================================================= */
function formatMarkdown(text) {
  if (!text) return '';

  // Normalize line endings
  text = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

  // ── Step 1: Extract fenced code blocks (protect from further processing) ──
  var codeBlocks = [];
  text = text.replace(/```[\w]*\n?([\s\S]*?)```/g, function(_, code) {
    var safe = code.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    codeBlocks.push('<pre class="md-code-block"><code>' + safe + '</code></pre>');
    return '\x02CB' + (codeBlocks.length - 1) + '\x02';
  });

  // ── Step 2: HTML-escape the remaining text ──
  text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // ── Step 3: Inline formatter — operates on already-escaped text ──
  function inline(t) {
    if (!t) return '';
    // Bold + italic
    t = t.replace(/\*\*\*([^*\n]+)\*\*\*/g, '<strong><em>$1</em></strong>');
    // Bold
    t = t.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
    // Italic (not preceded/followed by another *)
    t = t.replace(/(^|[^*])\*([^*\n]+)\*($|[^*])/g, '$1<em>$2</em>$3');
    // Inline code
    t = t.replace(/`([^`\n]+)`/g, '<code class="md-inline-code">$1</code>');
    return t;
  }

  // ── Step 4: Table builder ──
  function buildTable(rows) {
    var valid = rows.filter(function(r) { return r.trim(); });
    if (!valid.length) return '';
    var html = '<div class="md-table-wrap"><table class="md-table">';
    var headerDone = false;
    valid.forEach(function(row) {
      // Separator row  |---|---|  — marks end of header
      if (/^\|[\s\-:|]+\|$/.test(row.trim())) { headerDone = true; return; }
      var cells = row.split('|');
      cells = cells.slice(1, cells.length - 1);
      if (!headerDone) {
        html += '<thead><tr>' + cells.map(function(c) {
          return '<th>' + inline(c.trim()) + '</th>';
        }).join('') + '</tr></thead><tbody>';
        headerDone = true; // treat first row as header even without separator
      } else {
        html += '<tr>' + cells.map(function(c) {
          return '<td>' + inline(c.trim()) + '</td>';
        }).join('') + '</tr>';
      }
    });
    html += '</tbody></table></div>';
    return html;
  }

  // ── Step 5: Line-by-line block parser ──
  var lines = text.split('\n');
  var out = [];
  var i = 0;

  function isBlockStart(l) {
    if (!l || l.trim() === '') return true;
    if (/^\x02CB\d+\x02$/.test(l.trim())) return true;
    if (/^-{3,}$/.test(l.trim())) return true;
    if (/^#{1,6}\s/.test(l)) return true;
    if (l.trim().startsWith('|')) return true;
    if (/^\s*[-*\u2022]\s+/.test(l)) return true;
    if (/^\s*\d+[.):]\s+/.test(l)) return true;
    return false;
  }

  while (i < lines.length) {
    var line = lines[i];

    // Restored code block
    var cbMatch = line.trim().match(/^\x02CB(\d+)\x02$/);
    if (cbMatch) {
      out.push(codeBlocks[parseInt(cbMatch[1])]);
      i++; continue;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(line.trim()) || /^\*{3,}$/.test(line.trim())) {
      out.push('<hr class="md-hr">');
      i++; continue;
    }

    // Heading  # ## ### ####
    var hm = line.match(/^(#{1,6})\s+(.+)$/);
    if (hm) {
      var lvl = Math.min(hm[1].length, 4);
      out.push('<h' + (lvl + 1) + ' class="md-heading md-h' + lvl + '">' + inline(hm[2]) + '</h' + (lvl + 1) + '>');
      i++; continue;
    }

    // Table — collect consecutive pipe-starting lines
    if (line.trim().startsWith('|')) {
      var tbl = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) { tbl.push(lines[i]); i++; }
      out.push(buildTable(tbl));
      continue;
    }

    // Unordered list
    if (/^\s*[-*\u2022]\s+/.test(line)) {
      var uli = [];
      while (i < lines.length && /^\s*[-*\u2022]\s+/.test(lines[i])) {
        uli.push('<li>' + inline(lines[i].replace(/^\s*[-*\u2022]\s+/, '')) + '</li>');
        i++;
      }
      out.push('<ul class="md-list">' + uli.join('') + '</ul>');
      continue;
    }

    // Ordered list
    if (/^\s*\d+[.):,]\s+/.test(line)) {
      var oli = [];
      while (i < lines.length && /^\s*\d+[.):,]\s+/.test(lines[i])) {
        oli.push('<li>' + inline(lines[i].replace(/^\s*\d+[.):,]\s+/, '')) + '</li>');
        i++;
      }
      out.push('<ol class="md-list md-ol">' + oli.join('') + '</ol>');
      continue;
    }

    // Empty line — skip
    if (line.trim() === '') { i++; continue; }

    // Paragraph — collect consecutive non-block lines
    var para = [];
    while (i < lines.length && !isBlockStart(lines[i])) {
      para.push(inline(lines[i]));
      i++;
    }
    if (para.length) out.push('<p class="md-para">' + para.join('<br>') + '</p>');
  }

  return out.join('');
}

/* =============================================================
   13. INITIAL LOAD ANIMATIONS
   ============================================================= */
window.addEventListener('load', function() {
  window.scrollTo(0, 0);
  document.body.classList.add('loaded');

  if (typeof lucide !== 'undefined') {
    setTimeout(function() { lucide.createIcons(); }, 500);
  }
});
