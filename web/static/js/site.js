/* Shared responsive navigation. No external assets or trackers. */
(() => {
  const toggle = document.querySelector('.mobile-menu');
  const menu = document.getElementById('mobileNav');
  if (!toggle || !menu) return;
  toggle.addEventListener('click', () => {
    const isOpen = toggle.getAttribute('aria-expanded') === 'true';
    toggle.setAttribute('aria-expanded', String(!isOpen));
    menu.hidden = isOpen;
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !menu.hidden) {
      menu.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
      toggle.focus();
    }
  });
  document.addEventListener('click', e => {
    if (!menu.hidden && !menu.contains(e.target) && !toggle.contains(e.target)) {
      menu.hidden = true;
      toggle.setAttribute('aria-expanded', 'false');
    }
  });
})();
