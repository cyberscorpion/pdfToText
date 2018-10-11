from rest_framework.permissions import BasePermission
from rest_framework import permissions

from rest_framework.authentication import SessionAuthentication, BasicAuthentication 

class CsrfExemptSessionAuthentication(SessionAuthentication):

    def enforce_csrf(self, request):
        return